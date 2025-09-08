#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CCXT Crypto Alerts -> Telegram (auto-select Binance USDT pairs by liquidity)
v4: стабильная версия с:
- антиспамом (--cooldown_min, --only_change)
- фильтром тихих сетапов (--min_atr_pct)
- динамической точностью цен и форматами
- сопровождением позиций (--manage trail по EMA20)
- оценкой силы сетапа (score) в алертах и статистике
- paper-trade:
   * режим size = риск по стопу (default: --sizing risk --risk_pct 3.0)
   * альтернативно size = аллокация капитала (--sizing alloc с 3/6/9/15% от силы сетапа)
   * поддержка фьюч-логики: --futures_margin --leverage 100 (списывается маржа; Equity = bank+realized+unreal)
- трейдерская стата: Balance / Used / Free / Equity + Realized / Unrealized
- сохранение/восстановление состояния (paper_state.json, --reset_paper)
- исключение стейблов (USDC, FDUSD и т.п.) из базовой монеты
- порог минимального размера входа (--min_cost_usd)
- безопасные сообщения в Telegram (HTML-escape)
"""

import os, sys, time, argparse, re, json, html
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np

try:
    import ccxt
except Exception as e:
    print('Please install ccxt: pip install ccxt pandas numpy requests', file=sys.stderr)
    raise

try:
    import requests
except Exception:
    requests = None

# ===================== Indicators =====================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1 * delta.clip(upper=0.0)
    gain = up.ewm(alpha=1/length, min_periods=length).mean()
    loss = down.ewm(alpha=1/length, min_periods=length).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df['High']; low = df['Low']; close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, min_periods=length).mean()

# ===================== Config/Scoring =====================

@dataclass
class SignalConfig:
    lookback: int = 300
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    rsi_len: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    breakout_window: int = 20
    atr_len: int = 14
    risk_atr_mult: float = 1.5

def add_indicators(df: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    df = df.copy()
    df['ema20'] = ema(df['Close'], cfg.ema_fast)
    df['ema50'] = ema(df['Close'], cfg.ema_mid)
    df['ema200'] = ema(df['Close'], cfg.ema_slow)
    df['rsi'] = rsi(df['Close'], cfg.rsi_len)
    macd_line, signal_line, hist = macd(df['Close'], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = hist
    df['atr'] = atr(df, cfg.atr_len)
    df['vol_ma'] = df['Volume'].rolling(20).mean()
    return df.dropna().copy()

def confluence_score(direction: str, df: pd.DataFrame, cfg: SignalConfig) -> Tuple[int, List[str]]:
    reasons = []; score = 0
    close = df['Close']
    ema20 = df['ema20']; ema50 = df['ema50']; ema200 = df['ema200']
    rsi_now = df['rsi'].iloc[-1]
    macd_hist = df['macd_hist'].iloc[-1]
    macd_hist_slope = df['macd_hist'].iloc[-3:].diff().mean()
    vol = df['Volume']; vol_ma = df['vol_ma']
    atr_pct = df['atr'] / close * 100.0
    atr_now = atr_pct.iloc[-1]

    if direction == 'LONG':
        if (ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]):
            score += 20; reasons.append('EMA20>EMA50>EMA200 (uptrend)')
        elif close.iloc[-1] > ema200.iloc[-1]:
            score += 8; reasons.append('Price>EMA200')
        if rsi_now > 50: score += 10; reasons.append(f'RSI>50 ({rsi_now:.1f})')
        if macd_hist > 0: score += 10; reasons.append('MACD hist > 0')
        if macd_hist_slope > 0: score += 5; reasons.append('MACD momentum rising')
        recent_high = close.rolling(cfg.breakout_window).max().iloc[-2]
        if close.iloc[-1] > recent_high:
            score += 15; reasons.append(f'Breakout {cfg.breakout_window}-bar high')
        if close.iloc[-1] <= ema20.iloc[-1] and close.iloc[-1] >= ema50.iloc[-1]:
            score += 8; reasons.append('Pullback into EMA20-50')
    else:
        if (ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]):
            score += 20; reasons.append('EMA20<EMA50<EMA200 (downtrend)')
        elif close.iloc[-1] < ema200.iloc[-1]:
            score += 8; reasons.append('Price<EMA200')
        if rsi_now < 50: score += 10; reasons.append(f'RSI<50 ({rsi_now:.1f})')
        if macd_hist < 0: score += 10; reasons.append('MACD hist < 0')
        if macd_hist_slope < 0: score += 5; reasons.append('MACD momentum falling')
        recent_low = close.rolling(cfg.breakout_window).min().iloc[-2]
        if close.iloc[-1] < recent_low:
            score += 15; reasons.append(f'Breakdown {cfg.breakout_window}-bar low')
        if close.iloc[-1] >= ema20.iloc[-1] and close.iloc[-1] <= ema50.iloc[-1]:
            score += 8; reasons.append('Pullback into EMA20-50 (short)')

    if vol.iloc[-1] > 1.5 * vol_ma.iloc[-1]:
        score += 10; reasons.append('Volume spike >1.5x')

    if 1.0 <= atr_now <= 10.0:
        score += 7; reasons.append(f'ATR% sweet spot ({atr_now:.1f}%)')
    elif atr_now < 1.0:
        reasons.append(f'ATR% very low ({atr_now:.1f}%)')
    else:
        reasons.append(f'ATR% high ({atr_now:.1f}%)')

    return min(100, int(score)), reasons

# ===================== CCXT helpers =====================

SUPPORTED_TF = ['1m','3m','5m','15m','30m','1h','2h','4h','6h','12h','1d']

def ensure_exchange(name: str):
    name = name.lower()
    if not hasattr(ccxt, name):
        raise ValueError(f'Exchange {name} is not supported by ccxt')
    ex = getattr(ccxt, name)({'enableRateLimit': True})
    ex.load_markets()
    return ex


def with_retries(fn, *a, retries=3, base_delay=0.5, **kw):
    for i in range(retries + 1):
        try:
            return fn(*a, **kw)
        except Exception:
            if i == retries:
                raise
            time.sleep(base_delay * (2 ** i))


def fetch_ohlcv_paged(ex, symbol: str, timeframe: str, limit_total: int) -> List[List]:
    out = []; since = None
    per_call = 1000 if ex.has.get('fetchOHLCV', False) else 500
    while len(out) < limit_total:
        batch = with_retries(ex.fetch_ohlcv, symbol, timeframe=timeframe, since=since, limit=per_call)
        if not batch: break
        if out and batch[0][0] <= out[-1][0]:
            batch = [b for b in batch if b[0] > out[-1][0]]
        out.extend(batch)
        since = out[-1][0] + 1
        time.sleep(ex.rateLimit / 1000.0)
        if len(batch) < per_call: break
    return out[-limit_total:]

def fetch_history_ccxt(ex, symbol: str, timeframe: str, need_bars: int, allow_partial: bool, min_bars: int) -> Tuple[pd.DataFrame, int]:
    raw = fetch_ohlcv_paged(ex, symbol, timeframe, need_bars + 60)
    have = len(raw)
    if have < need_bars + 30 and not allow_partial:
        raise RuntimeError(f'Insufficient data {symbol} {timeframe}: got {have} bars')
    if have < min_bars:
        raise RuntimeError(f'Insufficient data (below min_bars={min_bars}) {symbol} {timeframe}: got {have} bars')
    raw = raw[-max(min(need_bars, have), min_bars):]
    df = pd.DataFrame(raw, columns=['Timestamp','Open','High','Low','Close','Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
    df = df.set_index('Timestamp')
    return df, have

EXCLUDE_PATTERNS = re.compile(r'(UP|DOWN|BULL|BEAR|.*3S|.*3L)', re.I)
STABLE_BASES = {
    'USDT','USDC','FDUSD','BUSD','TUSD','DAI','USDP','UST','USTC','USDD','PYUSD',
    'EUR','TRY','BRL','GBP','AUD','PAX'
}

def list_usdt_markets_by_volume(ex, min_quote_usd: float, limit: int, spot_only: bool = True) -> List[str]:
    if not ex.has.get('fetchTickers', False):
        raise RuntimeError('Exchange does not support fetchTickers for volume filtering.')
    tickers = with_retries(ex.fetch_tickers); rows = []
    for sym, t in tickers.items():
        try:
            market = ex.market(sym)
        except Exception:
            continue
        if market.get('quote') != 'USDT':
            continue
        if spot_only and market.get('type') not in (None, 'spot'):
            continue
        base = market.get('base', '')
        if EXCLUDE_PATTERNS.search(base): continue
        if base in STABLE_BASES: continue
        qv = 0.0
        if 'quoteVolume' in t and isinstance(t['quoteVolume'], (int,float)):
            qv = float(t['quoteVolume'])
        elif 'info' in t and isinstance(t['info'], dict):
            try: qv = float(t['info'].get('quoteVolume') or 0.0)
            except: qv = 0.0
        if qv >= min_quote_usd:
            rows.append((sym, qv))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in rows[:limit]]

# ===================== Telegram =====================

def tg_send(token: str, chat_id: str, text: str) -> bool:
    if requests is None:
        print('[warn] requests not installed; cannot send Telegram messages.', file=sys.stderr)
        return False
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    try:
        r = requests.post(url, json={'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML'}, timeout=20)
        ok = (r.status_code == 200) and r.json().get('ok', False)
        if not ok:
            print('[warn] Telegram send failed:', r.text[:200], file=sys.stderr)
        return ok
    except Exception as e:
        print('[warn] Telegram exception:', e, file=sys.stderr)
        return False


def tg_safe_send(token: str, chat_id: str, text: str) -> bool:
    if requests is None:
        print('[warn] requests not installed; cannot send Telegram messages.', file=sys.stderr)
        return False
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    try:
        r = requests.post(url, json={'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML'}, timeout=20)
        ok = (r.status_code == 200) and r.json().get('ok', False)
        if not ok:
            print('[warn] Telegram send failed:', r.text[:200], file=sys.stderr)
        time.sleep(0.35)
        return ok
    except Exception as e:
        print('[warn] Telegram exception:', e, file=sys.stderr)
        return False


def _decimals_for_price(p: float) -> int:
    if p <= 0: return 6
    if p < 0.0001: return 9
    if p < 0.001:  return 8
    if p < 0.01:   return 7
    if p < 0.1:    return 6
    return 6

def fmt_price(p: float) -> str:
    d = _decimals_for_price(abs(p))
    return f'{p:.{d}f}'

def fmt_qty(q: float) -> str:
    if q <= 0: return '0.000000'
    if q < 0.001: return f'{q:.8f}'
    if q < 1:     return f'{q:.6f}'
    if q < 100:   return f'{q:.4f}'
    return f'{q:.2f}'

def esc(s) -> str:
    try:
        return html.escape(str(s), quote=False)
    except Exception:
        return str(s)

def score_strength(score: int) -> str:
    if score >= 90:
        return "🔴 Экстра-сильный"
    elif score >= 80:
        return "🔵 Очень сильный"
    elif score >= 70:
        return "🟢 Сильный"
    elif score >= 60:
        return "⚪ Средний"
    else:
        return "▫️ Слабый"

def risk_pct_from_score(score: int) -> float:
    if score >= 90:
        return 15.0
    elif score >= 80:
        return 9.0
    elif score >= 70:
        return 6.0
    elif score >= 60:
        return 3.0
    else:
        return 1.0

def format_alert(exchange: str, symbol: str, tf: str, direction: str, score: int, price: float, reasons: List[str],
                 rsi_val: float, macd_hist: float, atr_pct: float, entry: float, stop: float, tp1: float, tp2: float,
                 partial_note: Optional[str], mgmt_note: Optional[str]=None) -> str:
    reasons_txt = '\n'.join('• ' + esc(r) for r in reasons)
    header = f'<b>{esc(exchange.upper())} {esc(symbol)}</b>  <code>{esc(tf)}</code>'
    if partial_note:
        header = f'⚠ {header}  <i>{esc(partial_note)}</i>'
    msg = (
        f'{header}\n'
        f'<b>{esc(direction)}</b> score: <b>{score}</b> ({score_strength(score)})  |  Price: <b>{fmt_price(price)}</b>\n'
        f'RSI: {rsi_val:.1f} | MACD hist: {macd_hist:.5f} | ATR%: {atr_pct:.2f}%\n'
        f'Entry: {fmt_price(entry)} | Stop: {fmt_price(stop)} | TP1: {fmt_price(tp1)} | TP2: {fmt_price(tp2)}\n'
        f'<i>Reasons:</i>\n{reasons_txt}'
    )
    if mgmt_note:
        msg += f'\n\n<b>Управление:</b>\n{esc(mgmt_note)}'
    return msg

# ===================== Explain (offline) / LLM =====================

def explain_signal(symbol: str, tf: str, d: Dict, df: pd.DataFrame) -> str:
    ema20, ema50, ema200 = df['ema20'].iloc[-1], df['ema50'].iloc[-1], df['ema200'].iloc[-1]
    if ema20 > ema50 > ema200: trend = "сильный аптренд"
    elif ema20 < ema50 < ema200: trend = "сильный даунтренд"
    else: trend = "смешанный/флэт"
    rsi_v, macd_h = d['rsi'], d['macd_hist']
    if rsi_v > 50 and macd_h > 0: mom = "бычий"
    elif rsi_v < 50 and macd_h < 0: mom = "медвежий"
    else: mom = "слабый/смешанный"
    atrp = d['atr_pct']
    if atrp < 1.0: vol_state = "очень низкая (риск пилы)"
    elif atrp <= 5.0: vol_state = "умеренная (комфортно)"
    else: vol_state = "высокая (быстрые движения)"
    close = df['Close'].iloc[-1]
    recent_high = df['Close'].rolling(20).max().iloc[-2]
    recent_low  = df['Close'].rolling(20).min().iloc[-2]
    if close > recent_high: structure = "пробой 20-барного хая"
    elif close < recent_low: structure = "пробой 20-барного лоя"
    else: structure = "внутри диапазона (ищем ретест)"
    vol, vol_ma = df['Volume'].iloc[-1], df['vol_ma'].iloc[-1]
    vol_note = "всплеск объёма" if vol > 1.5*vol_ma else "обычный объём"
    if d['direction']=="LONG":
        plan = f"Вход/ретест. Стоп {fmt_price(d['stop'])}. TP1 {fmt_price(d['tp1'])} (1R) → BE, TP2 {fmt_price(d['tp2'])}."
    else:
        plan = f"Шорт/ретест. Стоп {fmt_price(d['stop'])}. TP1 {fmt_price(d['tp1'])} (1R) → BE, TP2 {fmt_price(d['tp2'])}."
    lines = [
        f"Тренд: {trend} (EMA20={fmt_price(ema20)}, EMA50={fmt_price(ema50)}, EMA200={fmt_price(ema200)})",
        f"Моментум: {mom} (RSI={rsi_v:.1f}, MACD_hist={macd_h:.5f})",
        f"Волатильность: {vol_state} (ATR%={atrp:.2f}%)",
        f"Структура: {structure}",
        f"Объём: {vol_note}",
        "План: " + plan
    ]
    return "\n".join(lines)

def llm_explain(llm_url: str, llm_key: str, payload: Dict) -> str:
    if not requests or not llm_url: return ""
    try:
        headers = {'Content-Type':'application/json'}
        if llm_key: headers['Authorization'] = f'Bearer {llm_key}'
        prompt = f"Дай лаконичный трейдинг-разбор по пунктам на русском, без воды: {json.dumps(payload, ensure_ascii=False)}"
        r = requests.post(llm_url, headers=headers, json={'prompt': prompt}, timeout=25)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                if 'text' in data: return str(data['text'])
                if 'choices' in data and isinstance(data['choices'], list) and data['choices']:
                    ch0 = data['choices'][0]
                    if isinstance(ch0, dict) and 'text' in ch0: return str(ch0['text'])
        return ""
    except Exception:
        return ""

# ===================== Paper trading (spot/futures-like) =====================

@dataclass
class Position:
    symbol: str
    direction: str  # LONG/SHORT
    entry: float
    stop: float
    tp1: float
    tp2: float
    qty: float
    score: int = 0
    cost_initial: float = 0.0
    margin_initial: float = 0.0
    filled_tp1: bool = False
    closed: bool = False
    realized_pnl: float = 0.0

def entries_and_targets(direction: str, df: pd.DataFrame, cfg: SignalConfig) -> Tuple[float, float, float, float]:
    close = float(df['Close'].iloc[-1]); a = float(df['atr'].iloc[-1])
    if direction == 'LONG':
        entry = close; stop = close - cfg.risk_atr_mult * a
        tp1 = close + cfg.risk_atr_mult * a; tp2 = close + 2 * cfg.risk_atr_mult * a
    else:
        entry = close; stop = close + cfg.risk_atr_mult * a
        tp1 = close - cfg.risk_atr_mult * a; tp2 = close - 2 * cfg.risk_atr_mult * a
    return entry, stop, tp1, tp2

def simulate_fill(pos: Position, price: float) -> float:
    """Симуляция частичных выходов. Возвращает cash_delta для СПОТ (LONG). Для futures_margin — игнорируйте."""
    if pos.closed:
        return 0.0
    cash_delta = 0.0
    if pos.direction == 'LONG':
        if not pos.filled_tp1 and ((price >= pos.tp1) or (price <= pos.stop)):
            if price >= pos.tp1:
                qty_half = pos.qty * 0.5
                cash_delta += pos.tp1 * qty_half
                pnl = (pos.tp1 - pos.entry) * qty_half
                pos.realized_pnl += pnl
                pos.filled_tp1 = True
                pos.stop = pos.entry  # BE
            else:
                qty_all = pos.qty
                cash_delta += pos.stop * qty_all
                pnl = (pos.stop - pos.entry) * qty_all
                pos.realized_pnl += pnl
                pos.closed = True
                return cash_delta
        if pos.filled_tp1 and not pos.closed:
            live_qty = pos.qty * 0.5
            if price >= pos.tp2:
                cash_delta += pos.tp2 * live_qty
                pnl = (pos.tp2 - pos.entry) * live_qty
                pos.realized_pnl += pnl
                pos.closed = True
                return cash_delta
            if price <= pos.stop:
                cash_delta += pos.stop * live_qty
                pnl = (pos.stop - pos.entry) * live_qty
                pos.realized_pnl += pnl
                pos.closed = True
                return cash_delta
    else:
        # SHORT — для простоты считаем только PnL, кэш не двигаем
        if not pos.filled_tp1 and ((price <= pos.tp1) or (price >= pos.stop)):
            if price <= pos.tp1:
                pnl = (pos.entry - pos.tp1) * (pos.qty * 0.5)
                pos.realized_pnl += pnl
                pos.filled_tp1 = True
                pos.stop = pos.entry
            else:
                pnl = (pos.entry - pos.stop) * pos.qty
                pos.realized_pnl += pnl
                pos.closed = True
                return 0.0
        if pos.filled_tp1 and not pos.closed:
            if price <= pos.tp2:
                pnl = (pos.entry - pos.tp2) * (pos.qty * 0.5)
                pos.realized_pnl += pnl
                pos.closed = True
                return 0.0
            if price >= pos.stop:
                pnl = (pos.entry - pos.stop) * (pos.qty * 0.5)
                pos.realized_pnl += pnl
                pos.closed = True
                return 0.0
    return cash_delta

def unrealized_pnl(pos: Position, price: float) -> float:
    if pos.closed: return 0.0
    qty_live = pos.qty if not pos.filled_tp1 else pos.qty * 0.5
    if pos.direction == 'LONG':
        return (price - pos.entry) * qty_live
    else:
        return (pos.entry - price) * qty_live

# ===================== CLI/main =====================

DEFAULT_SYMBOLS = ['BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT','AVAX/USDT','XRP/USDT','ADA/USDT','DOGE/USDT','TON/USDT']

def main():
    parser = argparse.ArgumentParser(description='CCXT crypto alerts with auto USDT market selection (Binance)')
    parser.add_argument('--exchange', type=str, default='binance')
    parser.add_argument('--symbols', nargs='*', default=None)
    parser.add_argument('--auto_usdt', action='store_true')
    parser.add_argument('--min_quote_vol', type=float, default=5_000_000)
    parser.add_argument('--limit_pairs', type=int, default=30)
    parser.add_argument('--tf', nargs='*', default=['1h','4h'], choices=SUPPORTED_TF)
    parser.add_argument('--lookback', type=int, default=300)
    parser.add_argument('--min_bars', type=int, default=120)
    parser.add_argument('--min_score', type=int, default=60)
    parser.add_argument('--min_atr_pct', type=float, default=0.0, help='Filter out signals with ATR%% below this value')
    parser.add_argument('--mtf_confirm', action='store_true')
    parser.add_argument('--interval_sec', type=int, default=900)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--allow_partial', action='store_true')
    parser.add_argument('--explain', action='store_true')
    parser.add_argument('--llm_url', type=str, default='')
    parser.add_argument('--llm_key', type=str, default='')
    parser.add_argument('--cooldown_min', type=int, default=0, help='Anti-spam per symbol (minutes)')
    parser.add_argument('--only_change', action='store_true', help='Alert only when direction changes')
    parser.add_argument('--manage', type=str, default='off', choices=['off','trail'], help='Position management mode')
    # paper trading
    parser.add_argument('--paper_trade', action='store_true')
    parser.add_argument('--bank_start', type=float, default=10000.0)
    parser.add_argument('--sizing', type=str, default='risk', choices=['risk','alloc'], help='risk: фикс % риска по стопу; alloc: % аллокации капитала')
    parser.add_argument('--risk_pct', type=float, default=1.0, help='Процент риска от банка на сделку (для sizing=risk)')
    parser.add_argument('--leverage', type=float, default=1.0, help='Плечо (для учёта маржи при futures-режиме)')
    parser.add_argument('--futures_margin', action='store_true', help='Считать маржу как cost/leverage вместо списания всей стоимости (spot)')
    parser.add_argument('--min_cost_usd', type=float, default=10.0, help='Минимальная стоимость позиции/маржи для открытия (paper)')
    parser.add_argument('--tg_token', type=str, default=os.getenv('TELEGRAM_BOT_TOKEN',''))
    parser.add_argument('--tg_chat', type=str, default=os.getenv('TELEGRAM_CHAT_ID',''))
    parser.add_argument('--log_csv', type=str, default='ccxt_alerts_log.csv')
    parser.add_argument('--state_json', type=str, default='paper_state.json')
    parser.add_argument('--reset_paper', action='store_true')
    args = parser.parse_args()

    ex = ensure_exchange(args.exchange)

    # symbols
    if args.symbols:
        symbols = args.symbols
    elif args.auto_usdt:
        try:
            symbols = list_usdt_markets_by_volume(ex, args.min_quote_vol, args.limit_pairs, spot_only=True)
        except Exception as e:
            print(f'[warn] Auto list failed: {e}', file=sys.stderr)
            symbols = DEFAULT_SYMBOLS
        if not symbols:
            symbols = DEFAULT_SYMBOLS
        else:
            print(f'[info] Selected {len(symbols)} symbols: {", ".join(symbols)}')
    else:
        symbols = DEFAULT_SYMBOLS

    cfg = SignalConfig(lookback=args.lookback)
    last_dir: Dict[str, str] = {}
    last_alert_time: Dict[str, float] = {}
    positions: Dict[str, Position] = {}
    bank = args.bank_start
    equity = bank

    # ---------- Load/Save paper state ----------
    def save_state():
        try:
            data = {
                'bank': bank,
                'positions': [
                    {
                        'symbol': p.symbol,
                        'direction': p.direction,
                        'entry': p.entry,
                        'stop': p.stop,
                        'tp1': p.tp1,
                        'tp2': p.tp2,
                        'qty': p.qty,
                        'score': p.score,
                        'cost_initial': p.cost_initial,
                        'margin_initial': p.margin_initial,
                        'filled_tp1': p.filled_tp1,
                        'closed': p.closed,
                        'realized_pnl': p.realized_pnl,
                    } for p in positions.values()
                ]
            }
            with open(args.state_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f'[warn] save_state failed: {e}', file=sys.stderr)

    def load_state():
        nonlocal bank
        try:
            if args.reset_paper or not os.path.exists(args.state_json):
                return False, 0
            with open(args.state_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            bank = float(data.get('bank', bank))
            restored = 0
            for d in data.get('positions', []):
                p = Position(
                    symbol=d['symbol'],
                    direction=d['direction'],
                    entry=float(d['entry']),
                    stop=float(d['stop']),
                    tp1=float(d['tp1']),
                    tp2=float(d['tp2']),
                    qty=float(d['qty']),
                    score=int(d.get('score',0)),
                    cost_initial=float(d.get('cost_initial',0.0)),
                    margin_initial=float(d.get('margin_initial',0.0)),
                    filled_tp1=bool(d.get('filled_tp1', False)),
                    closed=bool(d.get('closed', False)),
                    realized_pnl=float(d.get('realized_pnl', 0.0)),
                )
                positions[p.symbol] = p
                restored += 1
            return True, restored
        except Exception as e:
            print(f'[warn] load_state failed: {e}', file=sys.stderr)
            return False, 0

    # ------------- helpers -------------
    def maybe_trail(pos: Position, df: pd.DataFrame):
        if args.manage != 'trail' or pos.closed: return None
        ema20 = float(df['ema20'].iloc[-1])
        note = None
        if pos.direction == 'LONG':
            new_stop = max(pos.stop, ema20)
            if new_stop > pos.stop:
                note = f'Trailing Stop: {fmt_price(pos.stop)} → {fmt_price(new_stop)} (EMA20)'
                pos.stop = new_stop
        else:
            new_stop = min(pos.stop, ema20)
            if new_stop < pos.stop:
                note = f'Trailing Stop: {fmt_price(pos.stop)} → {fmt_price(new_stop)} (EMA20)'
                pos.stop = new_stop
        return note

    def can_alert(sym: str, new_dir: str) -> bool:
        if args.only_change and last_dir.get(sym) == new_dir:
            return False
        if args.cooldown_min > 0:
            last = last_alert_time.get(sym, 0.0)
            if time.time() - last < args.cooldown_min * 60:
                return False
        return True

    def register_alert(sym: str, new_dir: str):
        last_dir[sym] = new_dir
        last_alert_time[sym] = time.time()

    def open_paper_position(sym: str, d: Dict):
        nonlocal bank
        rpct = risk_pct_from_score(int(d.get('score', 0)))  # сила сетапа для alloc
        entry = float(d['entry']); stop = float(d['stop'])
        if bank <= 0:
            return None
        if args.sizing == 'risk':
            # фикс. риск по стопу
            risk_usd = bank * (args.risk_pct/100.0)
            dist = abs(entry - stop)
            if dist <= 0:
                return None
            qty = risk_usd / dist
            cost = entry * qty
        else:
            # аллокация капитала
            cost = bank * (rpct/100.0)
            qty = cost / max(entry, 1e-12)

        if args.futures_margin:
            # маржа = cost/leverage
            required = cost / max(args.leverage, 1e-9)
            if required < args.min_cost_usd:
                return None
            if required > bank:
                scale = bank / required if required > 0 else 0.0
                qty *= scale
                cost = entry * qty
                required = bank
            if qty <= 0:
                return None
            bank -= required
            pos = Position(symbol=sym, direction=d['direction'], entry=entry, stop=stop,
                           tp1=d['tp1'], tp2=d['tp2'], qty=qty, score=int(d.get('score', 0)),
                           cost_initial=cost, margin_initial=required)
            positions[sym] = pos
            return pos, required, (args.risk_pct if args.sizing=='risk' else rpct)
        else:
            # спот: списываем полную стоимость
            if cost > bank:
                qty = max(0.0, bank / max(entry, 1e-12))
                cost = entry * qty
            if qty <= 0 or cost < args.min_cost_usd:
                return None
            bank -= cost
            pos = Position(symbol=sym, direction=d['direction'], entry=entry, stop=stop,
                           tp1=d['tp1'], tp2=d['tp2'], qty=qty, score=int(d.get('score', 0)),
                           cost_initial=cost, margin_initial=cost)
            positions[sym] = pos
            return pos, cost, (args.risk_pct if args.sizing=='risk' else rpct)

    def mark_and_stats(prices: Dict[str, float]):
        nonlocal equity, bank
        closed_count = 0
        # исполнение частичных выходов
        for sym, pos in list(positions.items()):
            price = prices.get(sym, None)
            if price is None: continue
            pre_closed = pos.closed
            cash_delta = simulate_fill(pos, price)
            if cash_delta != 0.0 and not args.futures_margin:
                bank += cash_delta  # на споте возвращаются деньги от частичных продаж
            if pos.closed and not pre_closed:
                closed_count += 1
                if args.futures_margin:
                    # освободить оставшуюся маржу
                    released = pos.margin_initial if not pos.filled_tp1 else pos.margin_initial*0.5
                    bank += released

        # пересчёт метрик
        used = 0.0
        unreal = 0.0
        for sym, pos in positions.items():
            if pos.closed: continue
            price = prices.get(sym)
            if price is None: continue
            live_qty = pos.qty if not pos.filled_tp1 else pos.qty*0.5
            if args.futures_margin:
                # used = занятая маржа
                used += pos.margin_initial if not pos.filled_tp1 else pos.margin_initial*0.5
                # unreal = mark-to-market
                if pos.direction == 'LONG':
                    unreal += (price - pos.entry) * live_qty
                else:
                    unreal += (pos.entry - price) * live_qty
            else:
                # spot: used = рыночная стоимость живой части, unreal = MV - cost
                mv = price * live_qty
                used += mv
                live_cost = pos.cost_initial if not pos.filled_tp1 else pos.cost_initial*0.5
                unreal += (mv - live_cost)

        realized = sum(p.realized_pnl for p in positions.values())
        if args.futures_margin:
            equity = bank + realized + unreal
        else:
            equity = bank + used
        free = max(0.0, bank)
        return unreal, closed_count, used, free

    def build_stats_text(realized: float, unreal: float, equity: float, used: float, free: float):
        open_lines = []
        open_scores = []
        for sym, pos in positions.items():
            if pos.closed: continue
            side = '🟢LONG' if pos.direction=='LONG' else '🔴SHORT'
            open_scores.append(pos.score)
            open_lines.append(
                f'{sym}: {side}, score {pos.score} ({score_strength(pos.score)}), '
                f'entry {fmt_price(pos.entry)}, stop {fmt_price(pos.stop)}, TP1 {fmt_price(pos.tp1)}, TP2 {fmt_price(pos.tp2)}'
            )
        avg_score_txt = ''
        if open_scores:
            avg = sum(open_scores)/len(open_scores)
            avg_score_txt = f" | Ср. score открытых: {avg:.1f} ({score_strength(int(avg))})"
        txt = (
            f'<b>Paper-trade</b>\n'
            f'Balance: ${bank:,.2f} | Used: ${used:,.2f} | Free: ${free:,.2f} | Equity: ${equity:,.2f}' + avg_score_txt + "\n"
            f'Realized: ${sum(p.realized_pnl for p in positions.values()):,.2f} | Unrealized: ${unreal:,.2f}\n'
        )
        if open_lines:
            txt += 'Открытые позиции:\n' + '\n'.join('• '+l for l in open_lines)
        else:
            txt += 'Открытых позиций нет.'
        return txt

    def score_signal_for_tf(ex, symbol: str, tf: str, cfg: SignalConfig,
                            allow_partial: bool, min_bars: int) -> Tuple[Dict, pd.DataFrame]:
        df, have_bars = fetch_history_ccxt(ex, symbol, tf, cfg.lookback, allow_partial, min_bars)
        df = add_indicators(df, cfg); df = df.iloc[-cfg.lookback:]
        long_score, long_reasons = confluence_score('LONG', df, cfg)
        short_score, short_reasons = confluence_score('SHORT', df, cfg)
        if long_score >= short_score:
            entry, stop, tp1, tp2 = entries_and_targets('LONG', df, cfg)
            d = {'direction':'LONG','score':long_score,'reasons':long_reasons,
                 'price':float(df['Close'].iloc[-1]), 'rsi':float(df['rsi'].iloc[-1]),
                 'macd_hist':float(df['macd_hist'].iloc[-1]),
                 'atr_pct':float((df['atr'].iloc[-1]/df['Close'].iloc[-1])*100.0),
                 'entry':entry,'stop':stop,'tp1':tp1,'tp2':tp2,
                 'bar_time':df.index[-1].isoformat(),'have_bars':have_bars,'need_bars':cfg.lookback}
        else:
            entry, stop, tp1, tp2 = entries_and_targets('SHORT', df, cfg)
            d = {'direction':'SHORT','score':short_score,'reasons':short_reasons,
                 'price':float(df['Close'].iloc[-1]), 'rsi':float(df['rsi'].iloc[-1]),
                 'macd_hist':float(df['macd_hist'].iloc[-1]),
                 'atr_pct':float((df['atr'].iloc[-1]/df['Close'].iloc[-1])*100.0),
                 'entry':entry,'stop':stop,'tp1':tp1,'tp2':tp2,
                 'bar_time':df.index[-1].isoformat(),'have_bars':have_bars,'need_bars':cfg.lookback}
        return d, df

    def cycle_once():
        alerts = 0
        prices: Dict[str, float] = {}
        for symbol in symbols:
            tf_scores = {}; tf_dfs = {}
            for tf in args.tf:
                try:
                    d, df = score_signal_for_tf(ex, symbol, tf, cfg, args.allow_partial, args.min_bars)
                except Exception as e:
                    print(f'[warn] {args.exchange} {symbol} {tf}: {e}', file=sys.stderr); continue
                if args.min_atr_pct > 0.0 and d['atr_pct'] < args.min_atr_pct:
                    continue
                tf_scores[tf] = d; tf_dfs[tf] = df
            if not tf_scores: continue

            # выбор сигнала
            if args.mtf_confirm:
                dirs: Dict[str, List[Tuple[str, Dict]]] = {}
                for tf, d in tf_scores.items():
                    if d['score'] >= args.min_score:
                        dirs.setdefault(d['direction'], []).append((tf, d))
                if not dirs: continue
                direction = max(dirs.items(), key=lambda kv: len(kv[1]))[0]
                tf_best, d_best = sorted(dirs[direction], key=lambda x: x[1]['score'], reverse=True)[0]
                df_best = tf_dfs[tf_best]
            else:
                tf_best, d_best = max(tf_scores.items(), key=lambda kv: kv[1]['score'])
                if d_best['score'] < args.min_score: continue
                df_best = tf_dfs[tf_best]

            prices[symbol] = d_best['price']

            # антиспам/смена направления
            if not can_alert(symbol, d_best['direction']):
                if symbol in positions and not positions[symbol].closed:
                    maybe_trail(positions[symbol], df_best)
                continue

            partial_note = None
            if d_best['have_bars'] < d_best['need_bars']:
                partial_note = f'bars: {d_best["have_bars"]}/{d_best["need_bars"]} (надёжность ниже)'

            mgmt_note = None
            if symbol in positions and not positions[symbol].closed:
                mgmt_note = maybe_trail(positions[symbol], df_best)

            text = format_alert(args.exchange, symbol, tf_best, d_best['direction'], d_best['score'], d_best['price'],
                                d_best['reasons'], d_best['rsi'], d_best['macd_hist'], d_best['atr_pct'],
                                d_best['entry'], d_best['stop'], d_best['tp1'], d_best['tp2'], partial_note, mgmt_note)

            if args.explain:
                text += f"\n\n<b>Разбор:</b>\n{esc(explain_signal(symbol, tf_best, d_best, df_best))}"
            if args.llm_url:
                extra = llm_explain(args.llm_url, args.llm_key, {'exchange':args.exchange,'symbol':symbol,'tf':tf_best, **d_best})
                if extra:
                    text += f"\n\n<b>ИИ-комментарий:</b>\n{esc(extra)}"

            if args.tg_token and args.tg_chat:
                tg_safe_send(args.tg_token, args.tg_chat, text)
            else:
                print(text)

            alerts += 1
            register_alert(symbol, d_best['direction'])

            # paper-trade: открыть позицию
            if args.paper_trade:
                opened = open_paper_position(symbol, d_best)
                if opened:
                    pos, paid, rpct = opened
                    open_msg = (f'📈 Paper: открыт {esc(pos.direction)} {esc(symbol)}, qty={fmt_qty(pos.qty)}, '
                                + (f'margin=${paid:,.2f}, lev={args.leverage:.0f}x' if args.futures_margin else f'cost=${paid:,.2f}')
                                + f', risk={rpct:.1f}%')
                else:
                    reason = 'недостаточно Free' if bank <= args.min_cost_usd else f'размер сделки слишком мал (&lt;${args.min_cost_usd:.0f})'
                    open_msg = f'⚠️ Paper: пропуск входа по {esc(symbol)} — {reason}'
                if args.tg_token and args.tg_chat:
                    tg_safe_send(args.tg_token, args.tg_chat, open_msg)
                else:
                    print(open_msg)

        # обновим статы
        unreal, closed_count, used, free = mark_and_stats(prices)
        if args.paper_trade:
            realized = sum(p.realized_pnl for p in positions.values())
            summary = build_stats_text(realized, unreal, equity, used, free)
            if args.tg_token and args.tg_chat:
                tg_safe_send(args.tg_token, args.tg_chat, summary)
            else:
                print(summary)

        return alerts

    restored_ok, restored_count = load_state()
    if restored_ok and args.paper_trade:
        start_msg = f'🔄 Восстановлено позиций: {restored_count}. Balance=${bank:,.2f}.'
        if args.tg_token and args.tg_chat: tg_safe_send(args.tg_token, args.tg_chat, start_msg)
        else: print(start_msg)

    if args.once:
        n = cycle_once()
        try: save_state()
        except: pass
        print(f'[info] Sent {n} alerts.')
        return

    try:
        while True:
            n = cycle_once()
            try: save_state()
            except: pass
            print(f'[info] Sent {n} alerts.')
            time.sleep(args.interval_sec)
    except KeyboardInterrupt:
        try: save_state()
        except: pass
        print('\n[info] Stopped.')

if __name__ == '__main__':
    main()

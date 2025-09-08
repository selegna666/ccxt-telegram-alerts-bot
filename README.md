# CCXT Telegram Alerts Bot (USDT, Binance)

Алерт-бот по крипте на CCXT с Telegram-уведомлениями и paper-торговлей.

## Возможности
- авто-подбор ликвидных пар USDT (исключая стейблы/левередж-токены)
- сигналы (RSI, MACD, EMA20/50/200, breakout, объём, ATR%)
- антиспам: `--cooldown_min`, `--only_change`
- фильтр «тихих» сетапов: `--min_atr_pct`
- сопровождение позы: `--manage trail` (стоп по EMA20)
- режим размера позиции:
  - **risk** (по умолчанию): фиксированный риск по стопу `--risk_pct`
  - **alloc**: аллокация части банка 3/6/9/15% по силе сетапа
- paper-trade (спот или фьючерсный стиль маржи):
  - `--futures_margin --leverage X` — списывается только маржа `cost/leverage`
- сохранение/восстановление стейта: `paper_state.json`, `--reset_paper`
- безопасные сообщения в Telegram (HTML-escape)

## Быстрый старт

### Установка
```bat
pip install ccxt pandas numpy requests

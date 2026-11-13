# RustWeatherML

A production-grade machine learning system for weather prediction built entirely in Rust. This project demonstrates the complete ML lifecycle from data collection to model monitoring, using Evcxr Jupyter kernel for interactive exploration.

---

## 🌍 Live Weather Predictions

> Auto-updated every 3 h via GitHub Actions | Last run: 2026-11-13 14:08 UTC



<!-- BEGIN:LIVE_PREDICTIONS -->
### 24-Hour, 48-Hour & 72-Hour Forecast

| City | Country | As of (local) | Current | +24h | +48h | +72h | Rain 24h | NWP precip | ML only | Confidence |
|------|---------|---------------|---------|------|------|------|----------|------------|---------|------------|
| London | 🇬🇧 | 23:00 | +4.5°C | +4.9°C | +6.8°C | +8.1°C |  14% | 0.0 mm (0h) |  96% | ±3.5°C |
| Berlin | 🇩🇪 | 23:00 | +3.7°C | +3.2°C | +4.0°C | +5.9°C |  12% | 0.0 mm (0h) |  71% | ±3.5°C |
| Oslo | 🇳🇴 | 23:00 | +2.5°C | -1.0°C | +0.2°C | +2.6°C |  25% | 0.1 mm (0h) |  65% | ±3.5°C |
| Tokyo | 🇯🇵 | 23:00 | +10.7°C | +11.6°C | +11.7°C | +12.6°C |   6% | 0.0 mm (0h) |  12% | ±3.5°C |
| Shanghai | 🇨🇳 | 23:00 | +14.5°C | +13.7°C | +13.4°C | +13.8°C |  67% | 1.4 mm (2h) |  81% | ±3.5°C |
| Chongqing | 🇨🇳 | 23:00 | +14.4°C | +14.7°C | +14.8°C | +15.6°C |  68% | 1.6 mm (3h) |  96% | ±3.5°C |
| Nanjing | 🇨🇳 | 23:00 | +12.2°C | +12.1°C | +12.4°C | +13.3°C |  89% | 6.6 mm (14h) |  65% | ±3.5°C |
| Dubai | 🇦🇪 | 23:00 | +26.9°C | +25.6°C | +25.6°C | +25.5°C |  13% | 0.0 mm (0h) |  81% | ±3.5°C |

> **How to read the rain columns.** `Rain 24h` is the **blended** probability that there will be any rain at all in the next 24 h, computed as `α·p_NWP + (1−α)·p_ML` with `α = 0.9`. The physical NWP signal dominates because the ML classifier was trained on a target (`precip_sum_next_24h > 0 mm`) that was 73.6 % positive in the training set and therefore has a positive prior bias. `NWP precip` shows the *actual* predicted rainfall in millimetres from Open-Meteo's numerical weather model, plus how many hours will have any precipitation. `ML only` shows the bagging-ensemble probability in isolation so you can see the drift. `+24h`, `+48h`, `+72h` come from dedicated Ridge (α=10) regressors trained in Notebook 05 on `temp_next_{24,48,72}h` with RMSE 3.5 / 4.5 / 5.1 °C. `Confidence` is ±1σ = ±RMSE of the 24 h test. `As of (local)` is the city's local timestamp of the last observed hour that anchors the forecast (end of the Open-Meteo past window, typically 23:00 of the previous day).

### Hourly Predictions (next 24 h, per city)

Click a city to expand. Each row is one hour; `temp` is our Ridge 24 h model rolled across the past 24 hours, `rain %` is the calibrated bagging probability, `precip` and `clouds` and `wind` come from Open-Meteo's own NWP forecast.

<details><summary><strong>🇬🇧 London</strong> — Cfb, Europe/London</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-11-13 00:00 | +6.1°C |  11% |   2% |  96% | 0.0 mm |   0% |  9.0 km/h | Clear |
| +2 | 2026-11-13 01:00 | +5.2°C |  11% |   2% |  96% | 0.0 mm |   0% |  5.0 km/h | Clear |
| +3 | 2026-11-13 02:00 | +4.7°C |  11% |   2% |  96% | 0.0 mm |   0% |  5.0 km/h | Clear |
| +4 | 2026-11-13 03:00 | +4.7°C |  11% |   2% |  96% | 0.0 mm |   0% |  5.4 km/h | Clear |
| +5 | 2026-11-13 04:00 | +4.4°C |  11% |   2% |  96% | 0.0 mm |   0% |  4.7 km/h | Clear |
| +6 | 2026-11-13 05:00 | +4.0°C |  11% |   2% |  96% | 0.0 mm |   0% |  4.0 km/h | Clear |
| +7 | 2026-11-13 06:00 | +4.2°C |  11% |   2% |  96% | 0.0 mm |   0% |  4.3 km/h | Clear |
| +8 | 2026-11-13 07:00 | +3.8°C |  11% |   2% |  96% | 0.0 mm |   0% |  4.7 km/h | Clear |
| +9 | 2026-11-13 08:00 | +2.9°C |  11% |   2% |  96% | 0.0 mm |   0% |  4.7 km/h | Clear |
| +10 | 2026-11-13 09:00 | +3.6°C |  11% |   2% |  96% | 0.0 mm |   0% |  6.1 km/h | Clear |
| +11 | 2026-11-13 10:00 | +4.4°C |  11% |   2% |  96% | 0.0 mm |   0% |  6.8 km/h | Clear |
| +12 | 2026-11-13 11:00 | +5.1°C |  11% |   2% |  96% | 0.0 mm |   0% |  6.8 km/h | Clear |
| +13 | 2026-11-13 12:00 | +5.7°C |  11% |   2% |  96% | 0.0 mm |   0% |  6.1 km/h | Clear |
| +14 | 2026-11-13 13:00 | +6.3°C |  11% |   2% |  96% | 0.0 mm |   0% |  8.3 km/h | Clear |
| +15 | 2026-11-13 14:00 | +6.6°C |  11% |   2% |  96% | 0.0 mm |   0% |  8.3 km/h | Clear |
| +16 | 2026-11-13 15:00 | +6.4°C |  11% |   2% |  96% | 0.0 mm |   0% |  8.3 km/h | Clear |
| +17 | 2026-11-13 16:00 | +6.6°C |  11% |   2% |  96% | 0.0 mm |   0% |  6.1 km/h | Clear |
| +18 | 2026-11-13 17:00 | +6.5°C |  11% |   2% |  96% | 0.0 mm |   0% |  4.0 km/h | Clear |
| +19 | 2026-11-13 18:00 | +6.2°C |  11% |   2% |  96% | 0.0 mm |   0% |  4.0 km/h | Clear |
| +20 | 2026-11-13 19:00 | +5.7°C |  11% |   2% |  96% | 0.0 mm |   0% |  2.9 km/h | Clear |
| +21 | 2026-11-13 20:00 | +5.5°C |  11% |   2% |  96% | 0.0 mm |   0% |  4.7 km/h | Clear |
| +22 | 2026-11-13 21:00 | +5.1°C |  11% |   2% |  96% | 0.0 mm |   0% |  4.7 km/h | Clear |
| +23 | 2026-11-13 22:00 | +5.1°C |  10% |   2% |  81% | 0.0 mm |   0% |  3.6 km/h | Clear |
| +24 | 2026-11-13 23:00 | +4.9°C |  11% |   2% |  96% | 0.0 mm |   0% |  4.0 km/h | Clear |

</details>

<details><summary><strong>🇩🇪 Berlin</strong> — Cfb, Europe/Berlin</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-11-13 00:00 | +3.2°C |   8% |   2% |  65% | 0.0 mm |  45% | 11.7 km/h | Cloudy |
| +2 | 2026-11-13 01:00 | +3.3°C |   8% |   2% |  65% | 0.0 mm |  33% | 13.3 km/h | Clear |
| +3 | 2026-11-13 02:00 | +3.2°C |   8% |   2% |  65% | 0.0 mm |  42% | 14.1 km/h | Cloudy |
| +4 | 2026-11-13 03:00 | +3.1°C |   8% |   2% |  65% | 0.0 mm |   8% | 15.5 km/h | Clear |
| +5 | 2026-11-13 04:00 | +3.1°C |   8% |   2% |  65% | 0.0 mm |   8% | 15.2 km/h | Clear |
| +6 | 2026-11-13 05:00 | +2.8°C |   8% |   2% |  65% | 0.0 mm |  25% | 15.8 km/h | Clear |
| +7 | 2026-11-13 06:00 | +2.6°C |   8% |   2% |  65% | 0.0 mm |  32% | 15.5 km/h | Clear |
| +8 | 2026-11-13 07:00 | +2.2°C |   8% |   2% |  65% | 0.0 mm |  21% | 16.6 km/h | Clear |
| +9 | 2026-11-13 08:00 | +2.2°C |   9% |   2% |  71% | 0.0 mm |  22% | 16.2 km/h | Clear |
| +10 | 2026-11-13 09:00 | +2.3°C |   9% |   2% |  71% | 0.0 mm |  40% | 16.6 km/h | Clear |
| +11 | 2026-11-13 10:00 | +2.7°C |   9% |   2% |  71% | 0.0 mm |  80% | 17.6 km/h | Cloudy |
| +12 | 2026-11-13 11:00 | +2.5°C |   8% |   2% |  65% | 0.0 mm |  47% | 19.1 km/h | Cloudy |
| +13 | 2026-11-13 12:00 | +2.9°C |   8% |   2% |  65% | 0.0 mm |  15% | 20.5 km/h | Clear |
| +14 | 2026-11-13 13:00 | +3.4°C |   8% |   2% |  65% | 0.0 mm |  49% | 20.9 km/h | Clear |
| +15 | 2026-11-13 14:00 | +4.3°C |   9% |   2% |  71% | 0.0 mm |  58% | 21.6 km/h | Cloudy |
| +16 | 2026-11-13 15:00 | +4.4°C |   8% |   2% |  65% | 0.0 mm |  80% | 22.0 km/h | Cloudy |
| +17 | 2026-11-13 16:00 | +4.4°C |   8% |   2% |  65% | 0.0 mm |  60% | 19.4 km/h | Cloudy |
| +18 | 2026-11-13 17:00 | +4.3°C |   8% |   2% |  65% | 0.0 mm |  62% | 20.5 km/h | Cloudy |
| +19 | 2026-11-13 18:00 | +4.2°C |   8% |   2% |  65% | 0.0 mm | 100% | 18.4 km/h | Rainy |
| +20 | 2026-11-13 19:00 | +4.5°C |   9% |   2% |  71% | 0.0 mm |  95% | 19.4 km/h | Cloudy |
| +21 | 2026-11-13 20:00 | +3.9°C |   8% |   2% |  65% | 0.0 mm |   7% | 17.6 km/h | Clear |
| +22 | 2026-11-13 21:00 | +3.6°C |   9% |   2% |  71% | 0.0 mm |  95% | 17.3 km/h | Cloudy |
| +23 | 2026-11-13 22:00 | +3.6°C |   9% |   2% |  71% | 0.0 mm |  16% | 15.5 km/h | Clear |
| +24 | 2026-11-13 23:00 | +3.2°C |   9% |   2% |  71% | 0.0 mm |  30% | 14.8 km/h | Clear |

</details>

<details><summary><strong>🇳🇴 Oslo</strong> — Dfb, Europe/Oslo</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-11-13 00:00 | +1.7°C |  39% |  35% |  71% | 0.1 mm |  96% | 18.4 km/h | Rainy |
| +2 | 2026-11-13 01:00 | +0.8°C |   9% |   2% |  71% | 0.0 mm |  92% | 21.2 km/h | Cloudy |
| +3 | 2026-11-13 02:00 | +0.4°C |   8% |   2% |  62% | 0.0 mm |  86% | 17.6 km/h | Cloudy |
| +4 | 2026-11-13 03:00 | -0.3°C |   8% |   2% |  62% | 0.0 mm |  79% | 16.2 km/h | Cloudy |
| +5 | 2026-11-13 04:00 | -0.8°C |   8% |   2% |  62% | 0.0 mm |  80% | 14.8 km/h | Cloudy |
| +6 | 2026-11-13 05:00 | -1.3°C |  10% |   2% |  81% | 0.0 mm |  62% | 14.8 km/h | Cloudy |
| +7 | 2026-11-13 06:00 | -1.7°C |  10% |   2% |  81% | 0.0 mm |  23% | 13.7 km/h | Clear |
| +8 | 2026-11-13 07:00 | -1.4°C |  10% |   2% |  81% | 0.0 mm |  57% | 12.6 km/h | Cloudy |
| +9 | 2026-11-13 08:00 | -1.9°C |  10% |   2% |  81% | 0.0 mm |  38% | 10.8 km/h | Clear |
| +10 | 2026-11-13 09:00 | -2.0°C |  10% |   2% |  81% | 0.0 mm |  22% | 12.6 km/h | Clear |
| +11 | 2026-11-13 10:00 | -1.9°C |  10% |   2% |  81% | 0.0 mm |  43% | 16.9 km/h | Clear |
| +12 | 2026-11-13 11:00 | -1.3°C |  10% |   2% |  81% | 0.0 mm |  38% | 19.1 km/h | Clear |
| +13 | 2026-11-13 12:00 | -0.7°C |  10% |   2% |  81% | 0.0 mm |  39% | 20.2 km/h | Clear |
| +14 | 2026-11-13 13:00 | -0.3°C |  10% |   2% |  81% | 0.0 mm |  22% | 16.9 km/h | Clear |
| +15 | 2026-11-13 14:00 | +0.0°C |   8% |   2% |  65% | 0.0 mm |  19% | 19.4 km/h | Clear |
| +16 | 2026-11-13 15:00 | -0.6°C |  10% |   2% |  81% | 0.0 mm |  23% | 15.8 km/h | Clear |
| +17 | 2026-11-13 16:00 | -1.0°C |  10% |   2% |  81% | 0.0 mm |  16% |  7.9 km/h | Clear |
| +18 | 2026-11-13 17:00 | -1.3°C |  10% |   2% |  81% | 0.0 mm |   4% | 10.8 km/h | Clear |
| +19 | 2026-11-13 18:00 | -1.0°C |  10% |   2% |  81% | 0.0 mm |   0% |  9.0 km/h | Clear |
| +20 | 2026-11-13 19:00 | -1.2°C |   9% |   2% |  71% | 0.0 mm |   0% |  5.8 km/h | Clear |
| +21 | 2026-11-13 20:00 | -0.2°C |   8% |   2% |  65% | 0.0 mm |   0% |  8.3 km/h | Clear |
| +22 | 2026-11-13 21:00 | -0.6°C |   8% |   2% |  65% | 0.0 mm |   0% |  3.2 km/h | Clear |
| +23 | 2026-11-13 22:00 | -0.4°C |   8% |   2% |  65% | 0.0 mm |   0% |  5.8 km/h | Clear |
| +24 | 2026-11-13 23:00 | -1.0°C |   8% |   2% |  65% | 0.0 mm |   0% |  6.1 km/h | Clear |

</details>

<details><summary><strong>🇯🇵 Tokyo</strong> — Cfa, Asia/Tokyo</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-11-13 00:00 | +11.0°C |   3% |   2% |  11% | 0.0 mm |   0% |  2.8 km/h | Clear |
| +2 | 2026-11-13 01:00 | +10.2°C |   3% |   2% |  12% | 0.0 mm |  13% |  2.6 km/h | Clear |
| +3 | 2026-11-13 02:00 | +9.7°C |   3% |   2% |  11% | 0.0 mm |  17% |  3.1 km/h | Clear |
| +4 | 2026-11-13 03:00 | +9.4°C |   3% |   2% |  12% | 0.0 mm |   5% |  3.3 km/h | Clear |
| +5 | 2026-11-13 04:00 | +9.2°C |   2% |   2% |   5% | 0.0 mm |  22% |  3.4 km/h | Clear |
| +6 | 2026-11-13 05:00 | +8.9°C |   2% |   2% |   5% | 0.0 mm |  17% |  3.2 km/h | Clear |
| +7 | 2026-11-13 06:00 | +8.9°C |   3% |   2% |  12% | 0.0 mm |  14% |  3.2 km/h | Clear |
| +8 | 2026-11-13 07:00 | +9.1°C |   2% |   2% |   5% | 0.0 mm |  12% |  1.5 km/h | Clear |
| +9 | 2026-11-13 08:00 | +10.1°C |   2% |   2% |   5% | 0.0 mm |   7% |  1.1 km/h | Clear |
| +10 | 2026-11-13 09:00 | +11.3°C |   3% |   2% |  12% | 0.0 mm |   0% |  1.0 km/h | Clear |
| +11 | 2026-11-13 10:00 | +12.9°C |   2% |   2% |   5% | 0.0 mm |   0% |  1.5 km/h | Clear |
| +12 | 2026-11-13 11:00 | +14.6°C |   2% |   2% |   5% | 0.0 mm |   0% |  1.6 km/h | Clear |
| +13 | 2026-11-13 12:00 | +16.7°C |   3% |   2% |  12% | 0.0 mm |   0% |  1.8 km/h | Clear |
| +14 | 2026-11-13 13:00 | +17.7°C |   3% |   2% |  12% | 0.0 mm |   0% |  2.3 km/h | Clear |
| +15 | 2026-11-13 14:00 | +18.1°C |   3% |   2% |  12% | 0.0 mm |   0% |  3.9 km/h | Clear |
| +16 | 2026-11-13 15:00 | +17.6°C |   3% |   2% |  12% | 0.0 mm |   0% |  3.7 km/h | Clear |
| +17 | 2026-11-13 16:00 | +16.7°C |   3% |   2% |  12% | 0.0 mm |   0% |  2.6 km/h | Clear |
| +18 | 2026-11-13 17:00 | +16.0°C |   3% |   2% |  12% | 0.0 mm |   0% |  2.9 km/h | Clear |
| +19 | 2026-11-13 18:00 | +15.0°C |   3% |   2% |  12% | 0.0 mm |   0% |  3.4 km/h | Clear |
| +20 | 2026-11-13 19:00 | +14.1°C |   3% |   2% |  12% | 0.0 mm |   0% |  3.6 km/h | Clear |
| +21 | 2026-11-13 20:00 | +13.5°C |   3% |   2% |  12% | 0.0 mm |   0% |  3.9 km/h | Clear |
| +22 | 2026-11-13 21:00 | +12.8°C |   3% |   2% |  12% | 0.0 mm |   0% |  3.8 km/h | Clear |
| +23 | 2026-11-13 22:00 | +12.3°C |   3% |   2% |  12% | 0.0 mm |   4% |  3.6 km/h | Clear |
| +24 | 2026-11-13 23:00 | +11.6°C |   3% |   2% |  12% | 0.0 mm |   3% |  3.7 km/h | Clear |

</details>

<details><summary><strong>🇨🇳 Shanghai</strong> — Cfa, Asia/Shanghai</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-11-13 00:00 | +12.1°C |  10% |   2% |  81% | 0.0 mm | 100% |  3.5 km/h | Cloudy |
| +2 | 2026-11-13 01:00 | +12.3°C |  62% |  60% |  81% | 0.4 mm | 100% |  6.5 km/h | Rainy |
| +3 | 2026-11-13 02:00 | +12.3°C |  10% |   2% |  81% | 0.0 mm | 100% |  5.6 km/h | Cloudy |
| +4 | 2026-11-13 03:00 | +12.1°C |  40% |  35% |  81% | 0.1 mm | 100% |  5.4 km/h | Rainy |
| +5 | 2026-11-13 04:00 | +12.3°C |  40% |  35% |  81% | 0.1 mm | 100% |  4.0 km/h | Rainy |
| +6 | 2026-11-13 05:00 | +12.3°C |  10% |   2% |  81% | 0.0 mm | 100% |  3.8 km/h | Cloudy |
| +7 | 2026-11-13 06:00 | +12.4°C |   8% |   2% |  65% | 0.0 mm | 100% |  4.4 km/h | Cloudy |
| +8 | 2026-11-13 07:00 | +13.3°C |   8% |   2% |  65% | 0.0 mm | 100% |  6.7 km/h | Cloudy |
| +9 | 2026-11-13 08:00 | +14.9°C |   8% |   2% |  65% | 0.0 mm | 100% |  5.5 km/h | Cloudy |
| +10 | 2026-11-13 09:00 | +16.0°C |   8% |   2% |  65% | 0.0 mm | 100% |  5.0 km/h | Cloudy |
| +11 | 2026-11-13 10:00 | +17.0°C |  38% |  35% |  65% | 0.1 mm | 100% |  6.3 km/h | Rainy |
| +12 | 2026-11-13 11:00 | +17.7°C |  61% |  60% |  65% | 0.6 mm | 100% |  5.0 km/h | Rainy |
| +13 | 2026-11-13 12:00 | +18.6°C |   8% |   2% |  65% | 0.0 mm | 100% |  6.2 km/h | Cloudy |
| +14 | 2026-11-13 13:00 | +18.8°C |   8% |   2% |  65% | 0.0 mm | 100% |  3.6 km/h | Cloudy |
| +15 | 2026-11-13 14:00 | +19.0°C |   8% |   2% |  65% | 0.0 mm | 100% |  1.5 km/h | Cloudy |
| +16 | 2026-11-13 15:00 | +18.5°C |   8% |   2% |  65% | 0.0 mm | 100% |  4.6 km/h | Cloudy |
| +17 | 2026-11-13 16:00 | +18.0°C |   8% |   2% |  65% | 0.0 mm | 100% |  6.1 km/h | Cloudy |
| +18 | 2026-11-13 17:00 | +16.7°C |  11% |   2% |  96% | 0.0 mm | 100% |  4.4 km/h | Cloudy |
| +19 | 2026-11-13 18:00 | +15.6°C |  11% |   2% |  96% | 0.0 mm | 100% |  5.6 km/h | Cloudy |
| +20 | 2026-11-13 19:00 | +15.2°C |  41% |  35% |  96% | 0.1 mm | 100% |  6.9 km/h | Rainy |
| +21 | 2026-11-13 20:00 | +15.0°C |  11% |   2% |  96% | 0.0 mm | 100% |  3.9 km/h | Cloudy |
| +22 | 2026-11-13 21:00 | +14.3°C |  11% |   2% |  96% | 0.0 mm | 100% |  5.1 km/h | Cloudy |
| +23 | 2026-11-13 22:00 | +13.9°C |  10% |   2% |  81% | 0.0 mm | 100% |  4.6 km/h | Cloudy |
| +24 | 2026-11-13 23:00 | +13.7°C |  10% |   2% |  81% | 0.0 mm | 100% |  5.0 km/h | Cloudy |

</details>

<details><summary><strong>🇨🇳 Chongqing</strong> — Cwa, Asia/Shanghai</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-11-13 00:00 | +13.7°C |   8% |   2% |  65% | 0.0 mm | 100% |  1.8 km/h | Cloudy |
| +2 | 2026-11-13 01:00 | +13.7°C |   8% |   2% |  65% | 0.0 mm | 100% |  2.3 km/h | Cloudy |
| +3 | 2026-11-13 02:00 | +13.7°C |   8% |   2% |  65% | 0.0 mm | 100% |  2.7 km/h | Cloudy |
| +4 | 2026-11-13 03:00 | +13.5°C |   8% |   2% |  65% | 0.0 mm | 100% |  0.8 km/h | Cloudy |
| +5 | 2026-11-13 04:00 | +13.5°C |   8% |   2% |  65% | 0.0 mm | 100% |  1.4 km/h | Cloudy |
| +6 | 2026-11-13 05:00 | +13.5°C |   8% |   2% |  65% | 0.0 mm | 100% |  1.5 km/h | Cloudy |
| +7 | 2026-11-13 06:00 | +13.4°C |  41% |  35% |  96% | 0.1 mm |  99% |  1.3 km/h | Rainy |
| +8 | 2026-11-13 07:00 | +13.7°C |  41% |  35% |  96% | 0.1 mm | 100% |  2.3 km/h | Rainy |
| +9 | 2026-11-13 08:00 | +14.6°C |  11% |   2% |  96% | 0.0 mm | 100% |  2.4 km/h | Cloudy |
| +10 | 2026-11-13 09:00 | +15.0°C |  41% |  35% |  96% | 0.1 mm | 100% |  2.2 km/h | Rainy |
| +11 | 2026-11-13 10:00 | +15.7°C |  64% |  60% |  96% | 0.3 mm | 100% |  3.5 km/h | Rainy |
| +12 | 2026-11-13 11:00 | +16.4°C |  64% |  60% |  96% | 0.7 mm | 100% |  5.1 km/h | Rainy |
| +13 | 2026-11-13 12:00 | +16.8°C |  41% |  35% |  96% | 0.2 mm | 100% |  5.9 km/h | Rainy |
| +14 | 2026-11-13 13:00 | +17.0°C |  41% |  35% |  96% | 0.1 mm | 100% |  5.3 km/h | Rainy |
| +15 | 2026-11-13 14:00 | +16.9°C |  11% |   2% |  96% | 0.0 mm | 100% |  3.8 km/h | Cloudy |
| +16 | 2026-11-13 15:00 | +17.1°C |  11% |   2% |  96% | 0.0 mm |  84% |  1.8 km/h | Cloudy |
| +17 | 2026-11-13 16:00 | +16.8°C |  11% |   2% |  96% | 0.0 mm |  42% |  2.6 km/h | Clear |
| +18 | 2026-11-13 17:00 | +16.4°C |  11% |   2% |  96% | 0.0 mm |  64% |  2.9 km/h | Cloudy |
| +19 | 2026-11-13 18:00 | +16.0°C |  11% |   2% |  96% | 0.0 mm |  53% |  2.8 km/h | Cloudy |
| +20 | 2026-11-13 19:00 | +15.7°C |  11% |   2% |  96% | 0.0 mm |  34% |  2.3 km/h | Clear |
| +21 | 2026-11-13 20:00 | +15.3°C |  11% |   2% |  96% | 0.0 mm |  21% |  2.5 km/h | Clear |
| +22 | 2026-11-13 21:00 | +15.1°C |  11% |   2% |  96% | 0.0 mm |  11% |  1.4 km/h | Clear |
| +23 | 2026-11-13 22:00 | +14.8°C |  11% |   2% |  96% | 0.0 mm |  11% |  0.6 km/h | Clear |
| +24 | 2026-11-13 23:00 | +14.7°C |  11% |   2% |  96% | 0.0 mm |  20% |  0.2 km/h | Clear |

</details>

<details><summary><strong>🇨🇳 Nanjing</strong> — Cfa, Asia/Shanghai</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-11-13 00:00 | +10.0°C |  61% |  60% |  65% | 0.4 mm | 100% |  5.3 km/h | Rainy |
| +2 | 2026-11-13 01:00 | +9.8°C |  38% |  35% |  65% | 0.2 mm | 100% |  5.0 km/h | Rainy |
| +3 | 2026-11-13 02:00 | +10.1°C |  38% |  35% |  65% | 0.2 mm | 100% |  7.0 km/h | Rainy |
| +4 | 2026-11-13 03:00 | +9.7°C |  61% |  60% |  65% | 0.5 mm | 100% |  5.2 km/h | Rainy |
| +5 | 2026-11-13 04:00 | +9.8°C |  83% |  85% |  65% | 1.3 mm | 100% |  6.1 km/h | Rainy |
| +6 | 2026-11-13 05:00 | +9.5°C |  61% |  60% |  65% | 0.5 mm | 100% |  4.8 km/h | Rainy |
| +7 | 2026-11-13 06:00 | +9.6°C |  38% |  35% |  65% | 0.1 mm | 100% |  3.4 km/h | Rainy |
| +8 | 2026-11-13 07:00 | +10.2°C |   8% |   2% |  65% | 0.0 mm |  99% |  2.5 km/h | Cloudy |
| +9 | 2026-11-13 08:00 | +11.4°C |   8% |   2% |  65% | 0.0 mm | 100% |  4.9 km/h | Cloudy |
| +10 | 2026-11-13 09:00 | +13.0°C |   8% |   2% |  65% | 0.0 mm | 100% |  4.7 km/h | Cloudy |
| +11 | 2026-11-13 10:00 | +14.5°C |   8% |   2% |  65% | 0.0 mm | 100% |  3.3 km/h | Cloudy |
| +12 | 2026-11-13 11:00 | +15.6°C |   8% |   2% |  65% | 0.0 mm |  99% |  2.3 km/h | Cloudy |
| +13 | 2026-11-13 12:00 | +16.1°C |   8% |   2% |  65% | 0.0 mm | 100% |  1.5 km/h | Cloudy |
| +14 | 2026-11-13 13:00 | +15.9°C |   8% |   2% |  65% | 0.0 mm | 100% |  6.9 km/h | Cloudy |
| +15 | 2026-11-13 14:00 | +15.7°C |  10% |   2% |  81% | 0.0 mm | 100% |  3.9 km/h | Cloudy |
| +16 | 2026-11-13 15:00 | +15.3°C |  40% |  35% |  81% | 0.1 mm | 100% |  4.1 km/h | Rainy |
| +17 | 2026-11-13 16:00 | +14.8°C |  62% |  60% |  81% | 0.4 mm | 100% |  2.1 km/h | Rainy |
| +18 | 2026-11-13 17:00 | +14.3°C |  61% |  60% |  65% | 0.5 mm | 100% |  1.4 km/h | Rainy |
| +19 | 2026-11-13 18:00 | +13.5°C |  62% |  60% |  81% | 0.5 mm | 100% |  2.7 km/h | Rainy |
| +20 | 2026-11-13 19:00 | +13.4°C |  62% |  60% |  81% | 0.4 mm | 100% |  4.9 km/h | Rainy |
| +21 | 2026-11-13 20:00 | +12.9°C |  64% |  60% |  96% | 0.5 mm | 100% |  5.3 km/h | Rainy |
| +22 | 2026-11-13 21:00 | +12.5°C |  62% |  60% |  81% | 0.4 mm | 100% |  7.0 km/h | Rainy |
| +23 | 2026-11-13 22:00 | +12.3°C |  62% |  60% |  81% | 0.4 mm | 100% |  6.7 km/h | Rainy |
| +24 | 2026-11-13 23:00 | +12.1°C |  40% |  35% |  81% | 0.2 mm | 100% |  6.0 km/h | Rainy |

</details>

<details><summary><strong>🇦🇪 Dubai</strong> — BWh, Asia/Dubai</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-11-13 00:00 | +25.9°C |  10% |   2% |  81% | 0.0 mm | 100% | 14.5 km/h | Clear |
| +2 | 2026-11-13 01:00 | +25.6°C |  10% |   2% |  81% | 0.0 mm | 100% | 13.3 km/h | Clear |
| +3 | 2026-11-13 02:00 | +25.2°C |  10% |   2% |  81% | 0.0 mm | 100% | 13.1 km/h | Clear |
| +4 | 2026-11-13 03:00 | +25.4°C |  11% |   2% |  96% | 0.0 mm | 100% | 13.0 km/h | Clear |
| +5 | 2026-11-13 04:00 | +25.3°C |  11% |   2% |  96% | 0.0 mm |  98% | 12.5 km/h | Clear |
| +6 | 2026-11-13 05:00 | +25.5°C |  11% |   2% |  96% | 0.0 mm |  65% | 12.4 km/h | Clear |
| +7 | 2026-11-13 06:00 | +25.5°C |  11% |   2% |  96% | 0.0 mm |  55% | 11.7 km/h | Clear |
| +8 | 2026-11-13 07:00 | +25.6°C |  11% |   2% |  96% | 0.0 mm |  43% | 12.7 km/h | Clear |
| +9 | 2026-11-13 08:00 | +26.2°C |  11% |   2% |  96% | 0.0 mm |  55% | 13.1 km/h | Clear |
| +10 | 2026-11-13 09:00 | +27.4°C |  11% |   2% |  96% | 0.0 mm |  43% | 13.9 km/h | Clear |
| +11 | 2026-11-13 10:00 | +28.1°C |  10% |   2% |  81% | 0.0 mm |  38% | 13.4 km/h | Clear |
| +12 | 2026-11-13 11:00 | +29.2°C |  10% |   2% |  81% | 0.0 mm |  40% | 13.8 km/h | Clear |
| +13 | 2026-11-13 12:00 | +29.9°C |  10% |   2% |  81% | 0.0 mm |  40% | 14.2 km/h | Clear |
| +14 | 2026-11-13 13:00 | +29.5°C |  10% |   2% |  81% | 0.0 mm |  37% | 15.5 km/h | Clear |
| +15 | 2026-11-13 14:00 | +29.1°C |   8% |   2% |  65% | 0.0 mm |  33% | 15.6 km/h | Clear |
| +16 | 2026-11-13 15:00 | +28.7°C |   8% |   2% |  65% | 0.0 mm |  26% | 15.7 km/h | Clear |
| +17 | 2026-11-13 16:00 | +28.1°C |  10% |   2% |  81% | 0.0 mm |  27% | 15.0 km/h | Clear |
| +18 | 2026-11-13 17:00 | +27.4°C |  10% |   2% |  81% | 0.0 mm |  28% | 14.0 km/h | Clear |
| +19 | 2026-11-13 18:00 | +26.9°C |  10% |   2% |  81% | 0.0 mm |  25% | 12.8 km/h | Clear |
| +20 | 2026-11-13 19:00 | +26.5°C |  10% |   2% |  81% | 0.0 mm |  22% | 12.0 km/h | Clear |
| +21 | 2026-11-13 20:00 | +26.3°C |  11% |   2% |  96% | 0.0 mm |  12% | 10.9 km/h | Clear |
| +22 | 2026-11-13 21:00 | +26.0°C |  10% |   2% |  81% | 0.0 mm |  14% | 10.7 km/h | Clear |
| +23 | 2026-11-13 22:00 | +25.7°C |  10% |   2% |  81% | 0.0 mm |  18% | 10.7 km/h | Clear |
| +24 | 2026-11-13 23:00 | +25.6°C |  11% |   2% |  96% | 0.0 mm |   8% | 10.6 km/h | Clear |

</details>


### Model Performance (held-out test set)

| Metric | Value |
|--------|-------|
| 24 h model | Ridge (alpha=10) |
| 48 h model | Ridge 48h (alpha=10) |
| 72 h model | Ridge 72h (alpha=10) |
| Rain model | Bagging ensemble (30 DT trees) + histogram calibration |
| Test RMSE (24 h) | 3.53 °C |
| Test RMSE (48 h) | 4.46 °C |
| Test RMSE (72 h) | 5.07 °C |
| Bias correction (24 h) | +0.69 °C |
| Contract version | 2.0.0 |
| Successful cities | 8 / 14 |

### Drift Monitor (last run)

⚠ Max PSI = 0.713  |  observations = 2352  |  cities = 14  |  status = drift detected

| Feature | PSI | KS | Mean shift (σ_ref) | Status |
|---------|-----|----|--------------------|--------|
| `clearness_index` | 0.089 | 0.069 | -0.17 | ok |
| `cloudcover` | 0.089 | 0.081 | +0.04 | ok |
| `dewpoint_2m` | 0.713 | 0.146 | -0.16 | severe |
| `precipitation` | 0.001 | 0.009 | -0.03 | ok |
| `pressure_msl` | 0.060 | 0.105 | +0.13 | ok |
| `relativehumidity_2m` | 0.017 | 0.030 | +0.08 | ok |
| `temp_lag24h` | 0.240 | 0.161 | -0.20 | severe |
| `temperature_2m` | 0.275 | 0.162 | -0.19 | severe |
| `vpd` | 0.052 | 0.079 | -0.19 | ok |
| `windspeed_10m` | 0.322 | 0.202 | -0.49 | severe |

<!-- END:LIVE_PREDICTIONS -->

---

## 📋 Project Overview

### Features

- **Complete ML Pipeline**: Data collection → Preprocessing → Feature Engineering → Model Training → Evaluation → Monitoring
- **Multiple ML Libraries**: Side-by-side comparison of linfa, smartcore, rustyml, and Burn
- **10 Years of Data**: Historical weather data from 2016-2025 for 14 cities worldwide
- **4 Prediction Tasks**:
  - Rain prediction (binary classification)
  - Weather condition classification (6 classes)
  - Temperature forecasting (24h, 48h, 72h)
  - Multi-target forecasting (temp + humidity + wind)
- **Drift Detection**: Automated monitoring for data and concept drift
- **Live Dashboard**: Daily auto-updated predictions via GitHub Actions

### Data Source

- **API**: [Open-Meteo](https://open-meteo.com/) (free, unlimited, no API key required)
- **Historical Data**: 2016-2025 (10 years)
- **Granularity**: Hourly observations
- **Total Records**: ~1.2 million

### Cities Covered


| Region              | Cities                                                   |
| ------------------- | -------------------------------------------------------- |
| Brazil 🇧🇷         | São Paulo, Rio de Janeiro, São José dos Campos, Campinas |
| USA 🇺🇸            | New York, Los Angeles                                    |
| Europe 🇬🇧🇩🇪🇳🇴 | London, Berlin, Oslo                                     |
| Asia 🇯🇵🇨🇳🇦🇪   | Tokyo, Shanghai, Chongqing, Nanjing, Dubai               |


---

## 🏗️ Project Structure

```
RustForMachineLearning/
├── notebooks/                    # Jupyter notebooks (Evcxr)
│   ├── 01_data_collection_and_exploration.ipynb
│   ├── 02_preprocessing_and_feature_engineering.ipynb
│   ├── 03_feature_selection_and_model_training.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   ├── 05_evaluation_and_validation.ipynb
│   └── 06_drift_detection_and_monitoring.ipynb
├── src/                          # Production Rust code
│   ├── lib.rs
│   ├── data/                     # Data loading & API client
│   ├── preprocessing/            # Data cleaning & transformation
│   ├── features/                 # Feature engineering & selection
│   ├── models/                   # ML model implementations
│   ├── training/                 # Training utilities
│   ├── evaluation/               # Metrics & visualization
│   ├── monitoring/               # Drift detection
│   └── bin/                      # CLI tools
├── data/                         # Data storage
│   ├── raw/                      # Raw API data
│   ├── processed/                # Cleaned data
│   └── features/                 # Engineered features
├── models/                       # Trained model artifacts
├── docs/                         # Documentation
└── .github/workflows/            # GitHub Actions
```

---

## 🚀 Getting Started

### Prerequisites

- Rust (1.70+)
- Jupyter with Evcxr kernel
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/SamoraDC/RustForMachineLearning.git
cd RustForMachineLearning

# Build the project
cargo build --release

# Install Evcxr Jupyter kernel (if not already installed)
cargo install evcxr_jupyter
evcxr_jupyter --install
```

### Running the Notebooks

```bash
# Start Jupyter
jupyter lab

# Navigate to notebooks/ and open the notebooks in order
```

### Running Daily Predictions

```bash
cargo run --release --bin daily_predictions
```

---

## 📊 ML Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    DATA     │───▶│  PREPROC    │───▶│  FEATURES   │
│ COLLECTION  │    │ & WRANGLING │    │ ENGINEERING │
└─────────────┘    └─────────────┘    └─────────────┘
       │                 │                   │
       ▼                 ▼                   ▼
  Open-Meteo API   Missing values      Lag features
  14 cities        Outliers            Rolling stats
  10 years         Normalization       Cyclical encoding

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  FEATURE    │───▶│   MODEL     │───▶│  TRAINING   │
│ SELECTION   │    │ COMPARISON  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
       │                 │                   │
       ▼                 ▼                   ▼
  Correlation       linfa            Train/Val/Test
  Importance        smartcore        Cross-validation
  Recursive elim    Burn             Early stopping

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   TUNING    │───▶│ EVALUATION  │───▶│ MONITORING  │
│             │    │             │    │ & DRIFT     │
└─────────────┘    └─────────────┘    └─────────────┘
       │                 │                   │
       ▼                 ▼                   ▼
  Grid search       Accuracy/F1       Data drift
  Random search     RMSE/MAE          Concept drift
  Cross-val         ROC/AUC           Auto-retrain
```

---

## 📈 Model Performance

### Classification Tasks


| Model               | Rain (Acc) | Rain (F1) | Condition (Acc) | Condition (F1) |
| ------------------- | ---------- | --------- | --------------- | -------------- |
| Logistic Regression | --         | --        | --              | --             |
| Decision Tree       | --         | --        | --              | --             |
| Random Forest       | --         | --        | --              | --             |
| Gradient Boosting   | --         | --        | --              | --             |
| Neural Network      | --         | --        | --              | --             |
| **Ensemble**        | --         | --        | --              | --             |


### Regression Tasks


| Model             | Temp 24h (RMSE) | Temp 48h (RMSE) | Temp 72h (RMSE) |
| ----------------- | --------------- | --------------- | --------------- |
| Linear Regression | --              | --              | --              |
| Decision Tree     | --              | --              | --              |
| Random Forest     | --              | --              | --              |
| Gradient Boosting | --              | --              | --              |
| Neural Network    | --              | --              | --              |
| **Ensemble**      | --              | --              | --              |


---

## 🔧 Technologies

### ML Libraries


| Library                                                | Purpose       | Status |
| ------------------------------------------------------ | ------------- | ------ |
| [linfa](https://github.com/rust-ml/linfa)              | Classical ML  | ✅      |
| [smartcore](https://github.com/smartcorelib/smartcore) | Classical ML  | ✅      |
| [rustyml](https://github.com/rustyml/rustyml)          | Classical ML  | 🔄     |
| [Burn](https://github.com/tracel-ai/burn)              | Deep Learning | 🔄     |


### Data & Utilities

- **polars**: DataFrame operations
- **ndarray**: N-dimensional arrays
- **reqwest**: HTTP client
- **chrono**: Date/time handling
- **plotters**: Visualization
- **statrs**: Statistical functions

---

## 📚 Documentation

- [Design Document](docs/plans/2026-01-23-rust-weather-ml-design.md)
- [API Reference](docs/api.md) *(coming soon)*
- [Contributing Guide](CONTRIBUTING.md) *(coming soon)*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Open-Meteo](https://open-meteo.com/) for providing free weather data
- [Rust ML Community](https://github.com/rust-ml) for the excellent ML libraries
- [Evcxr](https://github.com/evcxr/evcxr) for the Rust Jupyter kernel
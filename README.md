# RustWeatherML

A production-grade machine learning system for weather prediction built entirely in Rust. This project demonstrates the complete ML lifecycle from data collection to model monitoring, using Evcxr Jupyter kernel for interactive exploration.

---

## 🌍 Live Weather Predictions

> Auto-updated every 3 h via GitHub Actions | Last run: 2026-07-01 17:27 UTC



<!-- BEGIN:LIVE_PREDICTIONS -->
### 24-Hour, 48-Hour & 72-Hour Forecast

| City | Country | As of (local) | Current | +24h | +48h | +72h | Rain 24h | NWP precip | ML only | Confidence |
|------|---------|---------------|---------|------|------|------|----------|------------|---------|------------|
| Sao Paulo | 🇧🇷 | 23:00 | +18.5°C | +17.6°C | +17.7°C | +18.5°C |  14% | 0.0 mm (0h) |  96% | ±3.5°C |
| Rio de Janeiro | 🇧🇷 | 23:00 | +21.3°C | +21.4°C | +21.8°C | +22.5°C |  11% | 0.0 mm (0h) |  62% | ±3.5°C |
| Sao Jose dos Campos | 🇧🇷 | 23:00 | +15.3°C | +15.8°C | +16.7°C | +18.1°C |  12% | 0.0 mm (0h) |  71% | ±3.5°C |
| Campinas | 🇧🇷 | 23:00 | +18.9°C | +18.1°C | +18.9°C | +20.0°C |  14% | 0.0 mm (0h) |  96% | ±3.5°C |
| New York | 🇺🇸 | 23:00 | +26.8°C | +25.4°C | +24.8°C | +25.0°C |  13% | 0.0 mm (0h) |  81% | ±3.5°C |
| Los Angeles | 🇺🇸 | 23:00 | +17.6°C | +18.3°C | +19.1°C | +20.8°C |  13% | 0.0 mm (0h) |  81% | ±3.5°C |
| Berlin | 🇩🇪 | 23:00 | +21.6°C | +23.3°C | +22.6°C | +22.5°C |  47% | 0.5 mm (1h) |  65% | ±3.5°C |
| Tokyo | 🇯🇵 | 23:00 | +22.2°C | +23.9°C | +24.9°C | +25.8°C |  97% | 49.3 mm (14h) |  96% | ±3.5°C |
| Shanghai | 🇨🇳 | 23:00 | +23.5°C | +25.3°C | +26.6°C | +27.0°C |  91% | 6.1 mm (4h) |  81% | ±3.5°C |
| Chongqing | 🇨🇳 | 23:00 | +26.0°C | +26.7°C | +27.4°C | +28.1°C |  79% | 3.4 mm (4h) |  71% | ±3.5°C |
| Nanjing | 🇨🇳 | 23:00 | +25.8°C | +27.4°C | +28.3°C | +28.4°C |  50% | 0.8 mm (1h) |  96% | ±3.5°C |
| Dubai | 🇦🇪 | 23:00 | +32.1°C | +33.0°C | +34.1°C | +35.6°C |  12% | 0.0 mm (0h) |  71% | ±3.5°C |

> **How to read the rain columns.** `Rain 24h` is the **blended** probability that there will be any rain at all in the next 24 h, computed as `α·p_NWP + (1−α)·p_ML` with `α = 0.9`. The physical NWP signal dominates because the ML classifier was trained on a target (`precip_sum_next_24h > 0 mm`) that was 73.6 % positive in the training set and therefore has a positive prior bias. `NWP precip` shows the *actual* predicted rainfall in millimetres from Open-Meteo's numerical weather model, plus how many hours will have any precipitation. `ML only` shows the bagging-ensemble probability in isolation so you can see the drift. `+24h`, `+48h`, `+72h` come from dedicated Ridge (α=10) regressors trained in Notebook 05 on `temp_next_{24,48,72}h` with RMSE 3.5 / 4.5 / 5.1 °C. `Confidence` is ±1σ = ±RMSE of the 24 h test. `As of (local)` is the city's local timestamp of the last observed hour that anchors the forecast (end of the Open-Meteo past window, typically 23:00 of the previous day).

### Hourly Predictions (next 24 h, per city)

Click a city to expand. Each row is one hour; `temp` is our Ridge 24 h model rolled across the past 24 hours, `rain %` is the calibrated bagging probability, `precip` and `clouds` and `wind` come from Open-Meteo's own NWP forecast.

<details><summary><strong>🇧🇷 Sao Paulo</strong> — Cfa, America/Sao_Paulo</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-07-01 00:00 | +17.5°C |  11% |   2% |  96% | 0.0 mm |  63% |  9.0 km/h | Cloudy |
| +2 | 2026-07-01 01:00 | +16.8°C |  11% |   2% |  96% | 0.0 mm |   8% |  8.8 km/h | Clear |
| +3 | 2026-07-01 02:00 | +16.3°C |  11% |   2% |  96% | 0.0 mm |   0% |  9.2 km/h | Clear |
| +4 | 2026-07-01 03:00 | +16.0°C |  11% |   2% |  96% | 0.0 mm |   3% | 11.4 km/h | Clear |
| +5 | 2026-07-01 04:00 | +16.0°C |  11% |   2% |  96% | 0.0 mm |  17% | 11.8 km/h | Clear |
| +6 | 2026-07-01 05:00 | +15.6°C |  11% |   2% |  96% | 0.0 mm |   0% | 11.0 km/h | Clear |
| +7 | 2026-07-01 06:00 | +15.8°C |  11% |   2% |  96% | 0.0 mm |   0% | 10.4 km/h | Clear |
| +8 | 2026-07-01 07:00 | +15.2°C |  11% |   2% |  96% | 0.0 mm |   0% | 10.0 km/h | Clear |
| +9 | 2026-07-01 08:00 | +16.6°C |  11% |   2% |  96% | 0.0 mm |   0% | 10.2 km/h | Clear |
| +10 | 2026-07-01 09:00 | +18.6°C |  11% |   2% |  96% | 0.0 mm |   0% |  8.6 km/h | Clear |
| +11 | 2026-07-01 10:00 | +20.5°C |  11% |   2% |  96% | 0.0 mm |   0% |  9.4 km/h | Clear |
| +12 | 2026-07-01 11:00 | +22.3°C |  11% |   2% |  96% | 0.0 mm |   0% | 11.9 km/h | Clear |
| +13 | 2026-07-01 12:00 | +23.5°C |  11% |   2% |  96% | 0.0 mm |   0% | 11.4 km/h | Clear |
| +14 | 2026-07-01 13:00 | +24.4°C |  11% |   2% |  96% | 0.0 mm |   0% | 11.1 km/h | Clear |
| +15 | 2026-07-01 14:00 | +24.5°C |  11% |   2% |  96% | 0.0 mm |   0% | 12.5 km/h | Clear |
| +16 | 2026-07-01 15:00 | +24.7°C |  11% |   2% |  96% | 0.0 mm |   0% | 13.0 km/h | Clear |
| +17 | 2026-07-01 16:00 | +24.0°C |  11% |   2% |  96% | 0.0 mm |   0% | 12.6 km/h | Clear |
| +18 | 2026-07-01 17:00 | +22.9°C |  11% |   2% |  96% | 0.0 mm |   0% | 10.6 km/h | Clear |
| +19 | 2026-07-01 18:00 | +21.7°C |  11% |   2% |  96% | 0.0 mm |   0% |  8.6 km/h | Clear |
| +20 | 2026-07-01 19:00 | +20.7°C |  11% |   2% |  96% | 0.0 mm |   0% |  7.4 km/h | Clear |
| +21 | 2026-07-01 20:00 | +19.8°C |  11% |   2% |  96% | 0.0 mm |   0% |  5.3 km/h | Clear |
| +22 | 2026-07-01 21:00 | +18.9°C |  11% |   2% |  96% | 0.0 mm |   0% |  5.0 km/h | Clear |
| +23 | 2026-07-01 22:00 | +18.3°C |  11% |   2% |  96% | 0.0 mm |   0% |  6.7 km/h | Clear |
| +24 | 2026-07-01 23:00 | +17.6°C |  11% |   2% |  96% | 0.0 mm |   0% |  6.9 km/h | Clear |

</details>

<details><summary><strong>🇧🇷 Rio de Janeiro</strong> — Aw, America/Sao_Paulo</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-07-01 00:00 | +20.5°C |  10% |   2% |  81% | 0.0 mm |   7% |  4.3 km/h | Clear |
| +2 | 2026-07-01 01:00 | +20.5°C |   8% |   2% |  65% | 0.0 mm |   4% |  4.0 km/h | Clear |
| +3 | 2026-07-01 02:00 | +20.3°C |   8% |   2% |  65% | 0.0 mm |  32% |  5.6 km/h | Clear |
| +4 | 2026-07-01 03:00 | +19.9°C |   8% |   2% |  65% | 0.0 mm |  17% |  6.0 km/h | Clear |
| +5 | 2026-07-01 04:00 | +19.4°C |   8% |   2% |  65% | 0.0 mm |   4% |  7.8 km/h | Clear |
| +6 | 2026-07-01 05:00 | +19.5°C |   8% |   2% |  65% | 0.0 mm |   2% |  9.5 km/h | Clear |
| +7 | 2026-07-01 06:00 | +19.3°C |   8% |   2% |  65% | 0.0 mm |   1% | 10.1 km/h | Clear |
| +8 | 2026-07-01 07:00 | +19.6°C |   8% |   2% |  65% | 0.0 mm |   0% |  9.4 km/h | Clear |
| +9 | 2026-07-01 08:00 | +20.4°C |   8% |   2% |  65% | 0.0 mm |   5% |  7.4 km/h | Clear |
| +10 | 2026-07-01 09:00 | +22.1°C |   9% |   2% |  71% | 0.0 mm |  13% |  5.6 km/h | Clear |
| +11 | 2026-07-01 10:00 | +23.2°C |   8% |   2% |  65% | 0.0 mm |   0% |  5.9 km/h | Clear |
| +12 | 2026-07-01 11:00 | +24.9°C |   9% |   2% |  71% | 0.0 mm |   0% |  5.8 km/h | Clear |
| +13 | 2026-07-01 12:00 | +25.8°C |   8% |   2% |  65% | 0.0 mm |   0% |  7.0 km/h | Clear |
| +14 | 2026-07-01 13:00 | +26.4°C |   9% |   2% |  71% | 0.0 mm |   0% |  5.6 km/h | Clear |
| +15 | 2026-07-01 14:00 | +26.6°C |   5% |   2% |  35% | 0.0 mm |   0% |  4.6 km/h | Clear |
| +16 | 2026-07-01 15:00 | +26.3°C |   4% |   2% |  20% | 0.0 mm |   0% |  6.7 km/h | Clear |
| +17 | 2026-07-01 16:00 | +25.5°C |   4% |   2% |  20% | 0.0 mm |   0% |  7.0 km/h | Clear |
| +18 | 2026-07-01 17:00 | +24.3°C |   4% |   2% |  20% | 0.0 mm |   0% |  6.4 km/h | Clear |
| +19 | 2026-07-01 18:00 | +23.6°C |   3% |   2% |  11% | 0.0 mm |   0% |  3.8 km/h | Clear |
| +20 | 2026-07-01 19:00 | +23.3°C |   4% |   2% |  20% | 0.0 mm |   0% |  3.4 km/h | Clear |
| +21 | 2026-07-01 20:00 | +23.0°C |   5% |   2% |  35% | 0.0 mm |   0% |  2.9 km/h | Clear |
| +22 | 2026-07-01 21:00 | +22.9°C |   5% |   2% |  35% | 0.0 mm |   0% |  1.1 km/h | Clear |
| +23 | 2026-07-01 22:00 | +22.0°C |   5% |   2% |  35% | 0.0 mm |   0% |  1.7 km/h | Clear |
| +24 | 2026-07-01 23:00 | +21.4°C |   3% |   2% |  11% | 0.0 mm |   0% |  2.2 km/h | Clear |

</details>

<details><summary><strong>🇧🇷 Sao Jose dos Campos</strong> — Cfa, America/Sao_Paulo</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-07-01 00:00 | +16.7°C |   8% |   2% |  65% | 0.0 mm |  70% |  1.0 km/h | Cloudy |
| +2 | 2026-07-01 01:00 | +16.2°C |   8% |   2% |  65% | 0.0 mm |   1% |  0.4 km/h | Clear |
| +3 | 2026-07-01 02:00 | +15.8°C |   9% |   2% |  71% | 0.0 mm |  16% |  1.3 km/h | Clear |
| +4 | 2026-07-01 03:00 | +15.6°C |   8% |   2% |  65% | 0.0 mm | 100% |  1.5 km/h | Cloudy |
| +5 | 2026-07-01 04:00 | +15.6°C |   8% |   2% |  65% | 0.0 mm | 100% |  2.5 km/h | Cloudy |
| +6 | 2026-07-01 05:00 | +15.2°C |   8% |   2% |  65% | 0.0 mm | 100% |  2.7 km/h | Cloudy |
| +7 | 2026-07-01 06:00 | +15.1°C |   9% |   2% |  71% | 0.0 mm | 100% |  2.0 km/h | Cloudy |
| +8 | 2026-07-01 07:00 | +15.4°C |   8% |   2% |  65% | 0.0 mm | 100% |  1.2 km/h | Cloudy |
| +9 | 2026-07-01 08:00 | +15.6°C |   8% |   2% |  65% | 0.0 mm | 100% |  2.5 km/h | Cloudy |
| +10 | 2026-07-01 09:00 | +16.5°C |   8% |   2% |  65% | 0.0 mm |  98% |  2.8 km/h | Cloudy |
| +11 | 2026-07-01 10:00 | +18.9°C |   9% |   2% |  71% | 0.0 mm |   8% |  2.0 km/h | Clear |
| +12 | 2026-07-01 11:00 | +20.5°C |   9% |   2% |  71% | 0.0 mm |   2% |  1.7 km/h | Clear |
| +13 | 2026-07-01 12:00 | +22.4°C |   9% |   2% |  71% | 0.0 mm |   5% |  0.7 km/h | Clear |
| +14 | 2026-07-01 13:00 | +23.7°C |   9% |   2% |  71% | 0.0 mm |   5% |  2.1 km/h | Clear |
| +15 | 2026-07-01 14:00 | +24.4°C |   9% |   2% |  71% | 0.0 mm |   2% |  3.2 km/h | Clear |
| +16 | 2026-07-01 15:00 | +24.7°C |   9% |   2% |  71% | 0.0 mm |   1% |  5.0 km/h | Clear |
| +17 | 2026-07-01 16:00 | +24.0°C |   9% |   2% |  71% | 0.0 mm |   1% |  5.7 km/h | Clear |
| +18 | 2026-07-01 17:00 | +23.3°C |   5% |   2% |  35% | 0.0 mm |   0% |  4.0 km/h | Clear |
| +19 | 2026-07-01 18:00 | +21.0°C |   9% |   2% |  71% | 0.0 mm |   0% |  4.3 km/h | Clear |
| +20 | 2026-07-01 19:00 | +19.8°C |   8% |   2% |  62% | 0.0 mm |   0% |  3.0 km/h | Clear |
| +21 | 2026-07-01 20:00 | +18.5°C |   5% |   2% |  35% | 0.0 mm |   0% |  1.0 km/h | Clear |
| +22 | 2026-07-01 21:00 | +17.4°C |   8% |   2% |  62% | 0.0 mm |   0% |  0.5 km/h | Clear |
| +23 | 2026-07-01 22:00 | +16.2°C |   9% |   2% |  71% | 0.0 mm |   0% |  0.4 km/h | Clear |
| +24 | 2026-07-01 23:00 | +15.8°C |   8% |   2% |  62% | 0.0 mm |  98% |  0.4 km/h | Cloudy |

</details>

<details><summary><strong>🇧🇷 Campinas</strong> — Cfa, America/Sao_Paulo</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-07-01 00:00 | +18.3°C |  11% |   2% |  96% | 0.0 mm |   4% | 11.6 km/h | Clear |
| +2 | 2026-07-01 01:00 | +17.8°C |  11% |   2% |  96% | 0.0 mm |   3% | 10.9 km/h | Clear |
| +3 | 2026-07-01 02:00 | +17.0°C |  11% |   2% |  96% | 0.0 mm |   5% | 10.8 km/h | Clear |
| +4 | 2026-07-01 03:00 | +16.3°C |  11% |   2% |  96% | 0.0 mm |   0% | 11.1 km/h | Clear |
| +5 | 2026-07-01 04:00 | +16.4°C |  11% |   2% |  96% | 0.0 mm |   0% |  9.9 km/h | Clear |
| +6 | 2026-07-01 05:00 | +16.4°C |  11% |   2% |  96% | 0.0 mm |   0% | 10.7 km/h | Clear |
| +7 | 2026-07-01 06:00 | +16.9°C |  11% |   2% |  96% | 0.0 mm |   0% | 10.7 km/h | Clear |
| +8 | 2026-07-01 07:00 | +16.7°C |  10% |   2% |  81% | 0.0 mm |   0% |  9.8 km/h | Clear |
| +9 | 2026-07-01 08:00 | +18.4°C |  11% |   2% |  96% | 0.0 mm |   0% |  9.7 km/h | Clear |
| +10 | 2026-07-01 09:00 | +20.9°C |  11% |   2% |  96% | 0.0 mm |   0% |  8.2 km/h | Clear |
| +11 | 2026-07-01 10:00 | +22.2°C |  11% |   2% |  96% | 0.0 mm |   0% |  8.4 km/h | Clear |
| +12 | 2026-07-01 11:00 | +22.8°C |  10% |   2% |  81% | 0.0 mm |   0% |  8.9 km/h | Clear |
| +13 | 2026-07-01 12:00 | +24.0°C |  10% |   2% |  81% | 0.0 mm |   0% | 11.0 km/h | Clear |
| +14 | 2026-07-01 13:00 | +24.5°C |  10% |   2% |  81% | 0.0 mm |   0% | 11.8 km/h | Clear |
| +15 | 2026-07-01 14:00 | +24.6°C |   8% |   2% |  65% | 0.0 mm |   2% | 11.4 km/h | Clear |
| +16 | 2026-07-01 15:00 | +25.1°C |  10% |   2% |  81% | 0.0 mm |   2% | 10.6 km/h | Clear |
| +17 | 2026-07-01 16:00 | +25.0°C |  11% |   2% |  96% | 0.0 mm |   0% |  8.6 km/h | Clear |
| +18 | 2026-07-01 17:00 | +23.7°C |  11% |   2% |  96% | 0.0 mm |   0% |  4.7 km/h | Clear |
| +19 | 2026-07-01 18:00 | +21.6°C |  11% |   2% |  96% | 0.0 mm |   0% |  4.3 km/h | Clear |
| +20 | 2026-07-01 19:00 | +20.6°C |  11% |   2% |  96% | 0.0 mm |   1% |  2.7 km/h | Clear |
| +21 | 2026-07-01 20:00 | +19.6°C |  11% |   2% |  96% | 0.0 mm |   1% |  5.4 km/h | Clear |
| +22 | 2026-07-01 21:00 | +18.8°C |  11% |   2% |  96% | 0.0 mm |   0% |  6.4 km/h | Clear |
| +23 | 2026-07-01 22:00 | +18.5°C |  11% |   2% |  96% | 0.0 mm |   0% |  6.7 km/h | Clear |
| +24 | 2026-07-01 23:00 | +18.1°C |  11% |   2% |  96% | 0.0 mm |   0% |  7.3 km/h | Clear |

</details>

<details><summary><strong>🇺🇸 New York</strong> — Dfa, America/New_York</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-07-01 00:00 | +23.7°C |   8% |   2% |  65% | 0.0 mm |   2% | 14.8 km/h | Clear |
| +2 | 2026-07-01 01:00 | +23.3°C |   8% |   2% |  65% | 0.0 mm |   4% | 12.7 km/h | Clear |
| +3 | 2026-07-01 02:00 | +23.0°C |   8% |   2% |  65% | 0.0 mm |  50% | 15.7 km/h | Cloudy |
| +4 | 2026-07-01 03:00 | +22.4°C |  10% |   2% |  81% | 0.0 mm |   0% | 13.2 km/h | Clear |
| +5 | 2026-07-01 04:00 | +22.4°C |  10% |   2% |  81% | 0.0 mm |   0% | 16.0 km/h | Clear |
| +6 | 2026-07-01 05:00 | +22.0°C |  10% |   2% |  81% | 0.0 mm |   0% | 16.0 km/h | Clear |
| +7 | 2026-07-01 06:00 | +22.5°C |  10% |   2% |  81% | 0.0 mm |  11% |  8.0 km/h | Clear |
| +8 | 2026-07-01 07:00 | +24.3°C |  10% |   2% |  81% | 0.0 mm |   1% |  9.8 km/h | Clear |
| +9 | 2026-07-01 08:00 | +25.8°C |  10% |   2% |  81% | 0.0 mm |  13% | 11.7 km/h | Clear |
| +10 | 2026-07-01 09:00 | +27.3°C |  10% |   2% |  81% | 0.0 mm |  19% | 15.8 km/h | Clear |
| +11 | 2026-07-01 10:00 | +28.5°C |  10% |   2% |  81% | 0.0 mm |   5% | 14.8 km/h | Clear |
| +12 | 2026-07-01 11:00 | +30.4°C |  10% |   2% |  81% | 0.0 mm |   9% | 11.2 km/h | Clear |
| +13 | 2026-07-01 12:00 | +31.4°C |  10% |   2% |  81% | 0.0 mm |   0% | 11.2 km/h | Clear |
| +14 | 2026-07-01 13:00 | +31.9°C |  10% |   2% |  81% | 0.0 mm |   0% | 10.5 km/h | Clear |
| +15 | 2026-07-01 14:00 | +32.0°C |  10% |   2% |  81% | 0.0 mm |  23% | 10.9 km/h | Clear |
| +16 | 2026-07-01 15:00 | +31.2°C |  10% |   2% |  81% | 0.0 mm |  21% | 12.3 km/h | Clear |
| +17 | 2026-07-01 16:00 | +31.9°C |  11% |   2% |  96% | 0.0 mm |   0% | 16.5 km/h | Clear |
| +18 | 2026-07-01 17:00 | +31.8°C |  10% |   2% |  81% | 0.0 mm |   1% | 18.7 km/h | Clear |
| +19 | 2026-07-01 18:00 | +31.0°C |  10% |   2% |  81% | 0.0 mm |  13% | 17.3 km/h | Clear |
| +20 | 2026-07-01 19:00 | +30.8°C |  10% |   2% |  81% | 0.0 mm |   1% | 17.6 km/h | Clear |
| +21 | 2026-07-01 20:00 | +29.0°C |   8% |   2% |  65% | 0.0 mm |   0% | 16.3 km/h | Clear |
| +22 | 2026-07-01 21:00 | +26.5°C |  10% |   2% |  81% | 0.0 mm |   0% | 15.5 km/h | Clear |
| +23 | 2026-07-01 22:00 | +25.3°C |  10% |   2% |  81% | 0.0 mm |   0% | 15.1 km/h | Clear |
| +24 | 2026-07-01 23:00 | +25.4°C |   8% |   2% |  65% | 0.0 mm |   0% | 13.8 km/h | Clear |

</details>

<details><summary><strong>🇺🇸 Los Angeles</strong> — Csb, America/Los_Angeles</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-07-01 00:00 | +18.0°C |  10% |   2% |  81% | 0.0 mm | 100% |  4.6 km/h | Cloudy |
| +2 | 2026-07-01 01:00 | +18.2°C |  10% |   2% |  81% | 0.0 mm | 100% |  4.1 km/h | Cloudy |
| +3 | 2026-07-01 02:00 | +18.2°C |  10% |   2% |  81% | 0.0 mm | 100% |  5.7 km/h | Cloudy |
| +4 | 2026-07-01 03:00 | +18.2°C |  10% |   2% |  81% | 0.0 mm | 100% |  6.4 km/h | Cloudy |
| +5 | 2026-07-01 04:00 | +18.0°C |  10% |   2% |  81% | 0.0 mm | 100% |  4.0 km/h | Cloudy |
| +6 | 2026-07-01 05:00 | +18.3°C |  11% |   2% |  96% | 0.0 mm | 100% |  1.0 km/h | Cloudy |
| +7 | 2026-07-01 06:00 | +18.3°C |  11% |   2% |  96% | 0.0 mm | 100% |  1.0 km/h | Cloudy |
| +8 | 2026-07-01 07:00 | +18.8°C |  10% |   2% |  81% | 0.0 mm | 100% |  2.3 km/h | Cloudy |
| +9 | 2026-07-01 08:00 | +20.3°C |  10% |   2% |  81% | 0.0 mm | 100% |  1.8 km/h | Cloudy |
| +10 | 2026-07-01 09:00 | +21.0°C |  10% |   2% |  81% | 0.0 mm | 100% |  4.3 km/h | Cloudy |
| +11 | 2026-07-01 10:00 | +21.4°C |  10% |   2% |  81% | 0.0 mm | 100% |  5.2 km/h | Cloudy |
| +12 | 2026-07-01 11:00 | +22.3°C |  10% |   2% |  81% | 0.0 mm |  91% |  5.8 km/h | Cloudy |
| +13 | 2026-07-01 12:00 | +22.2°C |  10% |   2% |  81% | 0.0 mm |  46% |  7.1 km/h | Clear |
| +14 | 2026-07-01 13:00 | +23.3°C |  10% |   2% |  81% | 0.0 mm |  36% | 15.0 km/h | Clear |
| +15 | 2026-07-01 14:00 | +24.2°C |  10% |   2% |  81% | 0.0 mm |  14% | 15.6 km/h | Clear |
| +16 | 2026-07-01 15:00 | +24.0°C |  10% |   2% |  81% | 0.0 mm |  13% | 16.9 km/h | Clear |
| +17 | 2026-07-01 16:00 | +24.2°C |  10% |   2% |  81% | 0.0 mm |   7% | 16.5 km/h | Clear |
| +18 | 2026-07-01 17:00 | +22.9°C |  10% |   2% |  81% | 0.0 mm |   8% | 15.1 km/h | Clear |
| +19 | 2026-07-01 18:00 | +22.1°C |   8% |   2% |  65% | 0.0 mm |   8% | 14.2 km/h | Clear |
| +20 | 2026-07-01 19:00 | +21.0°C |  10% |   2% |  81% | 0.0 mm |  12% | 12.4 km/h | Clear |
| +21 | 2026-07-01 20:00 | +19.9°C |   8% |   2% |  65% | 0.0 mm |  10% | 10.0 km/h | Clear |
| +22 | 2026-07-01 21:00 | +19.2°C |   8% |   2% |  65% | 0.0 mm |  40% |  8.4 km/h | Clear |
| +23 | 2026-07-01 22:00 | +18.2°C |  10% |   2% |  81% | 0.0 mm |  59% |  7.0 km/h | Cloudy |
| +24 | 2026-07-01 23:00 | +18.3°C |   8% |   2% |  65% | 0.0 mm |  70% |  5.2 km/h | Cloudy |

</details>

<details><summary><strong>🇩🇪 Berlin</strong> — Cfb, Europe/Berlin</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-07-01 00:00 | +22.4°C |  11% |   2% |  96% | 0.0 mm | 100% |  7.1 km/h | Cloudy |
| +2 | 2026-07-01 01:00 | +21.8°C |  11% |   2% |  96% | 0.0 mm |  81% |  4.6 km/h | Cloudy |
| +3 | 2026-07-01 02:00 | +22.0°C |  11% |   2% |  96% | 0.0 mm | 100% |  4.8 km/h | Cloudy |
| +4 | 2026-07-01 03:00 | +20.9°C |  11% |   2% |  96% | 0.0 mm |  97% |  6.4 km/h | Cloudy |
| +5 | 2026-07-01 04:00 | +20.9°C |  11% |   2% |  96% | 0.0 mm | 100% |  6.9 km/h | Cloudy |
| +6 | 2026-07-01 05:00 | +19.8°C |  11% |   2% |  96% | 0.0 mm |  77% |  6.6 km/h | Cloudy |
| +7 | 2026-07-01 06:00 | +19.2°C |  11% |   2% |  96% | 0.0 mm |  95% |  5.8 km/h | Cloudy |
| +8 | 2026-07-01 07:00 | +19.6°C |  11% |   2% |  96% | 0.0 mm | 100% |  8.5 km/h | Cloudy |
| +9 | 2026-07-01 08:00 | +21.2°C |  11% |   2% |  96% | 0.0 mm | 100% |  7.2 km/h | Cloudy |
| +10 | 2026-07-01 09:00 | +23.0°C |  10% |   2% |  81% | 0.0 mm | 100% |  7.3 km/h | Cloudy |
| +11 | 2026-07-01 10:00 | +24.0°C |  61% |  60% |  65% | 0.3 mm | 100% |  7.9 km/h | Rainy |
| +12 | 2026-07-01 11:00 | +25.5°C |  40% |  35% |  81% | 0.1 mm | 100% |  7.4 km/h | Rainy |
| +13 | 2026-07-01 12:00 | +26.4°C |   8% |   2% |  65% | 0.0 mm | 100% |  9.0 km/h | Rainy |
| +14 | 2026-07-01 13:00 | +27.7°C |   8% |   2% |  65% | 0.0 mm | 100% | 10.5 km/h | Cloudy |
| +15 | 2026-07-01 14:00 | +28.8°C |   8% |   2% |  62% | 0.0 mm | 100% | 10.3 km/h | Cloudy |
| +16 | 2026-07-01 15:00 | +29.4°C |   8% |   2% |  62% | 0.0 mm | 100% | 10.1 km/h | Cloudy |
| +17 | 2026-07-01 16:00 | +29.6°C |   8% |   2% |  62% | 0.0 mm | 100% | 10.5 km/h | Cloudy |
| +18 | 2026-07-01 17:00 | +29.7°C |   8% |   2% |  62% | 0.0 mm | 100% | 11.0 km/h | Cloudy |
| +19 | 2026-07-01 18:00 | +28.3°C |   8% |   2% |  62% | 0.0 mm | 100% | 10.2 km/h | Cloudy |
| +20 | 2026-07-01 19:00 | +27.3°C |   8% |   2% |  62% | 0.0 mm | 100% |  9.7 km/h | Cloudy |
| +21 | 2026-07-01 20:00 | +26.1°C |   8% |   2% |  62% | 0.0 mm | 100% |  8.2 km/h | Cloudy |
| +22 | 2026-07-01 21:00 | +25.2°C |   9% |   2% |  71% | 0.0 mm | 100% |  9.3 km/h | Cloudy |
| +23 | 2026-07-01 22:00 | +24.6°C |  33% |  35% |  20% | 0.1 mm |  96% |  6.5 km/h | Rainy |
| +24 | 2026-07-01 23:00 | +23.3°C |   8% |   2% |  62% | 0.0 mm | 100% |  5.2 km/h | Cloudy |

</details>

<details><summary><strong>🇯🇵 Tokyo</strong> — Cfa, Asia/Tokyo</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-07-02 00:00 | +24.1°C |  64% |  60% |  96% | 0.3 mm | 100% |  4.1 km/h | Rainy |
| +2 | 2026-07-02 01:00 | +23.7°C |  41% |  35% |  96% | 0.1 mm |  85% |  4.0 km/h | Rainy |
| +3 | 2026-07-02 02:00 | +23.5°C |  64% |  60% |  96% | 0.5 mm |  76% |  3.6 km/h | Rainy |
| +4 | 2026-07-02 03:00 | +23.5°C |  97% |  97% |  96% | 3.1 mm |  80% |  2.3 km/h | Rainy |
| +5 | 2026-07-02 04:00 | +23.1°C |  97% |  97% |  96% | 7.2 mm |  88% |  6.2 km/h | Rainy |
| +6 | 2026-07-02 05:00 | +23.1°C |  97% |  97% |  96% | 9.3 mm |  90% |  4.4 km/h | Rainy |
| +7 | 2026-07-02 06:00 | +23.3°C |  97% |  97% |  96% | 11.7 mm |  82% |  5.9 km/h | Rainy |
| +8 | 2026-07-02 07:00 | +24.0°C |  97% |  97% |  96% | 7.7 mm |  69% |  5.4 km/h | Rainy |
| +9 | 2026-07-02 08:00 | +25.0°C |  86% |  85% |  96% | 2.0 mm |  79% |  5.8 km/h | Rainy |
| +10 | 2026-07-02 09:00 | +26.0°C |  97% |  97% |  96% | 4.9 mm |  76% |  3.3 km/h | Rainy |
| +11 | 2026-07-02 10:00 | +27.6°C |  86% |  85% |  96% | 1.2 mm |  97% |  4.4 km/h | Rainy |
| +12 | 2026-07-02 11:00 | +28.2°C |  64% |  60% |  96% | 0.3 mm |  90% |  4.3 km/h | Rainy |
| +13 | 2026-07-02 12:00 | +28.8°C |  41% |  35% |  96% | 0.2 mm | 100% |  3.3 km/h | Rainy |
| +14 | 2026-07-02 13:00 | +29.2°C |  41% |  35% |  96% | 0.2 mm |  94% |  2.2 km/h | Rainy |
| +15 | 2026-07-02 14:00 | +28.9°C |  64% |  60% |  96% | 0.5 mm | 100% |  2.1 km/h | Rainy |
| +16 | 2026-07-02 15:00 | +28.8°C |  11% |   2% |  96% | 0.0 mm |  87% |  4.7 km/h | Cloudy |
| +17 | 2026-07-02 16:00 | +28.7°C |  41% |  35% |  96% | 0.1 mm |  71% |  5.9 km/h | Rainy |
| +18 | 2026-07-02 17:00 | +27.4°C |  11% |   2% |  96% | 0.0 mm |  64% |  5.5 km/h | Cloudy |
| +19 | 2026-07-02 18:00 | +26.9°C |  11% |   2% |  96% | 0.0 mm |  90% |  4.4 km/h | Cloudy |
| +20 | 2026-07-02 19:00 | +25.9°C |  11% |   2% |  96% | 0.0 mm |  69% |  4.3 km/h | Cloudy |
| +21 | 2026-07-02 20:00 | +25.2°C |  11% |   2% |  96% | 0.0 mm |  51% |  4.0 km/h | Cloudy |
| +22 | 2026-07-02 21:00 | +25.0°C |  11% |   2% |  96% | 0.0 mm |  69% |  3.7 km/h | Cloudy |
| +23 | 2026-07-02 22:00 | +24.3°C |  11% |   2% |  96% | 0.0 mm |  46% |  3.4 km/h | Clear |
| +24 | 2026-07-02 23:00 | +23.9°C |  11% |   2% |  96% | 0.0 mm |  47% |  3.1 km/h | Clear |

</details>

<details><summary><strong>🇨🇳 Shanghai</strong> — Cfa, Asia/Shanghai</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-07-02 00:00 | +25.2°C |  11% |   2% |  96% | 0.0 mm |  70% |  6.1 km/h | Cloudy |
| +2 | 2026-07-02 01:00 | +25.0°C |  11% |   2% |  96% | 0.0 mm |  90% |  5.5 km/h | Cloudy |
| +3 | 2026-07-02 02:00 | +25.2°C |   8% |   2% |  62% | 0.0 mm |  95% |  6.0 km/h | Cloudy |
| +4 | 2026-07-02 03:00 | +25.0°C |   8% |   2% |  62% | 0.0 mm |  97% |  6.4 km/h | Cloudy |
| +5 | 2026-07-02 04:00 | +24.7°C |   8% |   2% |  62% | 0.0 mm | 100% |  6.3 km/h | Cloudy |
| +6 | 2026-07-02 05:00 | +24.5°C |   9% |   2% |  71% | 0.0 mm |  98% |  5.2 km/h | Cloudy |
| +7 | 2026-07-02 06:00 | +24.6°C |   9% |   2% |  71% | 0.0 mm |  98% |  5.4 km/h | Cloudy |
| +8 | 2026-07-02 07:00 | +26.0°C |   9% |   2% |  71% | 0.0 mm | 100% |  3.2 km/h | Cloudy |
| +9 | 2026-07-02 08:00 | +26.2°C |   9% |   2% |  71% | 0.0 mm | 100% |  4.8 km/h | Cloudy |
| +10 | 2026-07-02 09:00 | +27.0°C |  11% |   2% |  96% | 0.0 mm |  99% |  5.5 km/h | Cloudy |
| +11 | 2026-07-02 10:00 | +27.6°C |  11% |   2% |  96% | 0.0 mm | 100% |  7.3 km/h | Cloudy |
| +12 | 2026-07-02 11:00 | +27.5°C |  64% |  60% |  96% | 0.3 mm | 100% |  7.3 km/h | Rainy |
| +13 | 2026-07-02 12:00 | +27.0°C |  41% |  35% |  96% | 0.1 mm | 100% |  8.2 km/h | Rainy |
| +14 | 2026-07-02 13:00 | +27.5°C |   9% |   2% |  71% | 0.0 mm | 100% |  6.6 km/h | Cloudy |
| +15 | 2026-07-02 14:00 | +27.9°C |  41% |  35% |  96% | 0.1 mm | 100% |  6.9 km/h | Rainy |
| +16 | 2026-07-02 15:00 | +28.1°C |  11% |   2% |  96% | 0.0 mm | 100% |  8.0 km/h | Cloudy |
| +17 | 2026-07-02 16:00 | +28.4°C |  11% |   2% |  96% | 0.0 mm | 100% |  9.0 km/h | Cloudy |
| +18 | 2026-07-02 17:00 | +28.1°C |  41% |  35% |  96% | 0.1 mm | 100% | 10.0 km/h | Rainy |
| +19 | 2026-07-02 18:00 | +27.4°C |  86% |  85% |  96% | 2.3 mm | 100% |  8.8 km/h | Rainy |
| +20 | 2026-07-02 19:00 | +26.8°C |  86% |  85% |  96% | 1.9 mm | 100% |  7.0 km/h | Rainy |
| +21 | 2026-07-02 20:00 | +26.1°C |  86% |  85% |  96% | 1.0 mm | 100% |  6.4 km/h | Rainy |
| +22 | 2026-07-02 21:00 | +26.0°C |  41% |  35% |  96% | 0.1 mm | 100% |  7.2 km/h | Rainy |
| +23 | 2026-07-02 22:00 | +25.6°C |  41% |  35% |  96% | 0.1 mm | 100% |  5.5 km/h | Rainy |
| +24 | 2026-07-02 23:00 | +25.3°C |  41% |  35% |  96% | 0.1 mm | 100% |  6.5 km/h | Rainy |

</details>

<details><summary><strong>🇨🇳 Chongqing</strong> — Cwa, Asia/Shanghai</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-07-02 00:00 | +25.6°C |  41% |  35% |  96% | 0.1 mm | 100% |  3.1 km/h | Rainy |
| +2 | 2026-07-02 01:00 | +25.3°C |  11% |   2% |  96% | 0.0 mm |  82% |  1.5 km/h | Cloudy |
| +3 | 2026-07-02 02:00 | +25.7°C |  11% |   2% |  96% | 0.0 mm |  75% |  0.6 km/h | Cloudy |
| +4 | 2026-07-02 03:00 | +25.1°C |  11% |   2% |  96% | 0.0 mm |  75% |  1.0 km/h | Cloudy |
| +5 | 2026-07-02 04:00 | +24.9°C |  11% |   2% |  96% | 0.0 mm |  95% |  0.0 km/h | Cloudy |
| +6 | 2026-07-02 05:00 | +24.7°C |  11% |   2% |  96% | 0.0 mm |  61% |  0.0 km/h | Cloudy |
| +7 | 2026-07-02 06:00 | +25.2°C |  11% |   2% |  96% | 0.0 mm |  95% |  1.2 km/h | Cloudy |
| +8 | 2026-07-02 07:00 | +25.6°C |  11% |   2% |  96% | 0.0 mm | 100% |  1.3 km/h | Cloudy |
| +9 | 2026-07-02 08:00 | +26.5°C |  11% |   2% |  96% | 0.0 mm |  55% |  1.5 km/h | Cloudy |
| +10 | 2026-07-02 09:00 | +26.8°C |  11% |   2% |  96% | 0.0 mm |  86% |  1.0 km/h | Cloudy |
| +11 | 2026-07-02 10:00 | +27.7°C |  11% |   2% |  96% | 0.0 mm |  84% |  3.1 km/h | Cloudy |
| +12 | 2026-07-02 11:00 | +28.6°C |  11% |   2% |  96% | 0.0 mm | 100% |  4.2 km/h | Cloudy |
| +13 | 2026-07-02 12:00 | +29.1°C |  64% |  60% |  96% | 0.6 mm |  88% | 10.1 km/h | Stormy |
| +14 | 2026-07-02 13:00 | +29.3°C |  64% |  60% |  96% | 0.8 mm |  73% |  4.5 km/h | Rainy |
| +15 | 2026-07-02 14:00 | +30.2°C |   5% |   2% |  35% | 0.0 mm |  97% |  5.1 km/h | Cloudy |
| +16 | 2026-07-02 15:00 | +30.4°C |  35% |  35% |  35% | 0.1 mm | 100% |  6.8 km/h | Rainy |
| +17 | 2026-07-02 16:00 | +29.8°C |  38% |  35% |  62% | 0.1 mm |  90% |  5.9 km/h | Rainy |
| +18 | 2026-07-02 17:00 | +29.9°C |  32% |  35% |   5% | 0.1 mm |  48% |  5.1 km/h | Rainy |
| +19 | 2026-07-02 18:00 | +29.8°C |  32% |  35% |   5% | 0.1 mm |  38% |  4.8 km/h | Rainy |
| +20 | 2026-07-02 19:00 | +29.1°C |   2% |   2% |   5% | 0.0 mm |  96% |  3.6 km/h | Cloudy |
| +21 | 2026-07-02 20:00 | +28.7°C |  32% |  35% |   5% | 0.1 mm | 100% |  1.1 km/h | Rainy |
| +22 | 2026-07-02 21:00 | +27.9°C |  55% |  60% |  11% | 0.8 mm |  99% |  0.8 km/h | Stormy |
| +23 | 2026-07-02 22:00 | +27.5°C |  58% |  60% |  35% | 0.5 mm |  89% |  4.1 km/h | Stormy |
| +24 | 2026-07-02 23:00 | +26.7°C |  38% |  35% |  62% | 0.1 mm |  98% |  2.7 km/h | Rainy |

</details>

<details><summary><strong>🇨🇳 Nanjing</strong> — Cfa, Asia/Shanghai</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-07-02 00:00 | +25.2°C |  11% |   2% |  96% | 0.0 mm |  44% |  6.4 km/h | Clear |
| +2 | 2026-07-02 01:00 | +25.0°C |  11% |   2% |  96% | 0.0 mm |  57% |  6.3 km/h | Cloudy |
| +3 | 2026-07-02 02:00 | +25.2°C |  11% |   2% |  96% | 0.0 mm |   4% |  8.2 km/h | Clear |
| +4 | 2026-07-02 03:00 | +24.9°C |  11% |   2% |  96% | 0.0 mm |   4% |  6.9 km/h | Clear |
| +5 | 2026-07-02 04:00 | +24.6°C |  11% |   2% |  96% | 0.0 mm |  77% |  6.1 km/h | Cloudy |
| +6 | 2026-07-02 05:00 | +24.7°C |  11% |   2% |  96% | 0.0 mm |  48% |  7.6 km/h | Clear |
| +7 | 2026-07-02 06:00 | +25.0°C |  11% |   2% |  96% | 0.0 mm |  95% |  7.3 km/h | Cloudy |
| +8 | 2026-07-02 07:00 | +26.0°C |  11% |   2% |  96% | 0.0 mm |  99% |  6.1 km/h | Cloudy |
| +9 | 2026-07-02 08:00 | +26.5°C |  11% |   2% |  96% | 0.0 mm |  99% |  9.8 km/h | Cloudy |
| +10 | 2026-07-02 09:00 | +27.7°C |  11% |   2% |  96% | 0.0 mm |  98% |  8.7 km/h | Cloudy |
| +11 | 2026-07-02 10:00 | +28.9°C |  11% |   2% |  96% | 0.0 mm |  76% |  7.7 km/h | Cloudy |
| +12 | 2026-07-02 11:00 | +30.2°C |  41% |  35% |  96% | 0.1 mm |  87% |  7.0 km/h | Rainy |
| +13 | 2026-07-02 12:00 | +30.6°C |  41% |  35% |  96% | 0.1 mm |  95% |  8.0 km/h | Rainy |
| +14 | 2026-07-02 13:00 | +30.9°C |  41% |  35% |  96% | 0.1 mm |  59% |  8.9 km/h | Rainy |
| +15 | 2026-07-02 14:00 | +31.2°C |  64% |  60% |  96% | 0.4 mm |  71% | 12.8 km/h | Rainy |
| +16 | 2026-07-02 15:00 | +30.9°C |  41% |  35% |  96% | 0.1 mm |  19% | 10.1 km/h | Rainy |
| +17 | 2026-07-02 16:00 | +31.0°C |  11% |   2% |  96% | 0.0 mm |  34% | 10.6 km/h | Clear |
| +18 | 2026-07-02 17:00 | +30.5°C |  11% |   2% |  96% | 0.0 mm |  74% | 11.2 km/h | Cloudy |
| +19 | 2026-07-02 18:00 | +29.9°C |  11% |   2% |  96% | 0.0 mm |  59% | 11.6 km/h | Cloudy |
| +20 | 2026-07-02 19:00 | +29.1°C |  11% |   2% |  96% | 0.0 mm |  85% | 10.6 km/h | Cloudy |
| +21 | 2026-07-02 20:00 | +28.2°C |  11% |   2% |  96% | 0.0 mm |  99% |  9.6 km/h | Cloudy |
| +22 | 2026-07-02 21:00 | +27.6°C |  11% |   2% |  96% | 0.0 mm | 100% |  9.2 km/h | Cloudy |
| +23 | 2026-07-02 22:00 | +27.7°C |  11% |   2% |  96% | 0.0 mm | 100% |  7.7 km/h | Cloudy |
| +24 | 2026-07-02 23:00 | +27.4°C |  11% |   2% |  96% | 0.0 mm |  99% |  8.0 km/h | Cloudy |

</details>

<details><summary><strong>🇦🇪 Dubai</strong> — BWh, Asia/Dubai</summary>

| +h | local time | temp | rain % | (NWP) | (ML) | precip | clouds | wind | conditions |
|----|------------|------|--------|-------|------|--------|--------|------|------------|
| +1 | 2026-07-01 00:00 | +32.1°C |  10% |   2% |  81% | 0.0 mm |   0% |  7.8 km/h | Clear |
| +2 | 2026-07-01 01:00 | +31.8°C |   9% |   2% |  71% | 0.0 mm |   0% |  7.6 km/h | Clear |
| +3 | 2026-07-01 02:00 | +31.3°C |   9% |   2% |  71% | 0.0 mm |   0% |  7.1 km/h | Clear |
| +4 | 2026-07-01 03:00 | +30.8°C |   8% |   2% |  65% | 0.0 mm |  18% |  6.9 km/h | Clear |
| +5 | 2026-07-01 04:00 | +30.6°C |  10% |   2% |  81% | 0.0 mm |  93% |  6.3 km/h | Cloudy |
| +6 | 2026-07-01 05:00 | +30.6°C |   9% |   2% |  71% | 0.0 mm |  79% |  6.5 km/h | Cloudy |
| +7 | 2026-07-01 06:00 | +30.9°C |   9% |   2% |  71% | 0.0 mm |  57% |  7.9 km/h | Cloudy |
| +8 | 2026-07-01 07:00 | +32.0°C |   9% |   2% |  71% | 0.0 mm |  97% |  7.9 km/h | Cloudy |
| +9 | 2026-07-01 08:00 | +33.7°C |   8% |   2% |  65% | 0.0 mm |  71% |  7.5 km/h | Cloudy |
| +10 | 2026-07-01 09:00 | +35.1°C |   8% |   2% |  65% | 0.0 mm |  15% |  8.2 km/h | Clear |
| +11 | 2026-07-01 10:00 | +36.5°C |   9% |   2% |  71% | 0.0 mm |  29% |  9.9 km/h | Clear |
| +12 | 2026-07-01 11:00 | +36.5°C |   9% |   2% |  71% | 0.0 mm |  33% |  9.8 km/h | Clear |
| +13 | 2026-07-01 12:00 | +36.5°C |   8% |   2% |  65% | 0.0 mm |  45% | 10.5 km/h | Clear |
| +14 | 2026-07-01 13:00 | +37.1°C |   9% |   2% |  71% | 0.0 mm |  77% | 12.3 km/h | Cloudy |
| +15 | 2026-07-01 14:00 | +37.3°C |   8% |   2% |  65% | 0.0 mm |  97% | 13.1 km/h | Cloudy |
| +16 | 2026-07-01 15:00 | +37.3°C |   8% |   2% |  65% | 0.0 mm |  90% | 12.4 km/h | Cloudy |
| +17 | 2026-07-01 16:00 | +37.1°C |   9% |   2% |  71% | 0.0 mm |  94% | 11.5 km/h | Cloudy |
| +18 | 2026-07-01 17:00 | +35.8°C |   9% |   2% |  71% | 0.0 mm |  77% | 11.3 km/h | Cloudy |
| +19 | 2026-07-01 18:00 | +34.8°C |   8% |   2% |  62% | 0.0 mm |  65% | 10.0 km/h | Cloudy |
| +20 | 2026-07-01 19:00 | +33.9°C |   8% |   2% |  62% | 0.0 mm |   0% |  8.2 km/h | Clear |
| +21 | 2026-07-01 20:00 | +33.8°C |   8% |   2% |  62% | 0.0 mm |   0% |  6.5 km/h | Clear |
| +22 | 2026-07-01 21:00 | +33.5°C |   8% |   2% |  62% | 0.0 mm |   0% |  6.6 km/h | Clear |
| +23 | 2026-07-01 22:00 | +33.6°C |   8% |   2% |  62% | 0.0 mm |   0% |  7.9 km/h | Clear |
| +24 | 2026-07-01 23:00 | +33.0°C |   8% |   2% |  62% | 0.0 mm |   0% |  8.7 km/h | Clear |

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
| Successful cities | 12 / 14 |

### Drift Monitor (last run)

⚠ Max PSI = 2.937  |  observations = 2352  |  cities = 14  |  status = drift detected

| Feature | PSI | KS | Mean shift (σ_ref) | Status |
|---------|-----|----|--------------------|--------|
| `clearness_index` | 0.031 | 0.060 | +0.06 | ok |
| `cloudcover` | 0.071 | 0.122 | -0.21 | ok |
| `dewpoint_2m` | 2.536 | 0.286 | +0.57 | severe |
| `precipitation` | 0.002 | 0.013 | -0.01 | ok |
| `pressure_msl` | 0.212 | 0.101 | +0.04 | severe |
| `relativehumidity_2m` | 0.052 | 0.068 | -0.14 | ok |
| `temp_lag24h` | 2.937 | 0.361 | +0.63 | severe |
| `temperature_2m` | 2.775 | 0.338 | +0.61 | severe |
| `vpd` | 0.172 | 0.181 | +0.36 | moderate |
| `windspeed_10m` | 0.311 | 0.190 | -0.47 | severe |

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
# RustWeatherML

A production-grade machine learning system for weather prediction built entirely in Rust. This project demonstrates the complete ML lifecycle from data collection to model monitoring, using Evcxr Jupyter kernel for interactive exploration.

---

## 🌍 Live Weather Predictions

> Auto-updated daily at 06:00 UTC | Last run: 2026-04-15 15:37 UTC

### 24-Hour, 48-Hour & 72-Hour Forecast

| City | Country | Now (local) | Current | +24h | +48h | +72h | Rain 24h | Confidence |
|------|---------|-------------|---------|------|------|------|----------|------------|
| Sao Paulo | 🇧🇷 | 23:00 | +18.3°C | +19.0°C | +19.0°C | +19.3°C |  96% | ±3.5°C |
| Rio de Janeiro | 🇧🇷 | 23:00 | +22.1°C | +22.8°C | +22.8°C | +23.1°C |  71% | ±3.5°C |
| Sao Jose dos Campos | 🇧🇷 | 23:00 | +18.0°C | +18.4°C | +18.9°C | +19.4°C |  71% | ±3.5°C |
| Campinas | 🇧🇷 | 23:00 | +18.8°C | +19.6°C | +19.6°C | +20.0°C |  96% | ±3.5°C |
| New York | 🇺🇸 | 23:00 | +24.5°C | +21.2°C | +20.0°C | +19.8°C |  65% | ±3.5°C |
| Los Angeles | 🇺🇸 | 23:00 | +14.6°C | +16.1°C | +16.5°C | +17.4°C |  81% | ±3.5°C |
| London | 🇬🇧 | 23:00 | +14.5°C | +14.4°C | +14.3°C | +15.3°C |  96% | ±3.5°C |
| Berlin | 🇩🇪 | 23:00 | +9.4°C | +12.0°C | +13.6°C | +14.7°C |  81% | ±3.5°C |
| Oslo | 🇳🇴 | 23:00 | +8.7°C | +9.6°C | +10.3°C | +11.0°C |  81% | ±3.5°C |
| Tokyo | 🇯🇵 | 23:00 | +12.3°C | +11.9°C | +12.7°C | +14.1°C |  96% | ±3.5°C |
| Shanghai | 🇨🇳 | 23:00 | +15.3°C | +15.8°C | +16.1°C | +16.7°C |  96% | ±3.5°C |
| Chongqing | 🇨🇳 | 23:00 | +17.3°C | +18.5°C | +18.9°C | +19.3°C |  96% | ±3.5°C |
| Nanjing | 🇨🇳 | 23:00 | +16.3°C | +16.5°C | +16.5°C | +17.0°C |  96% | ±3.5°C |
| Dubai | 🇦🇪 | 23:00 | +24.3°C | +24.9°C | +25.3°C | +25.6°C |  96% | ±3.5°C |

> **Source of each horizon.** `+24h`, `+48h`, and `+72h` all come from dedicated Ridge (alpha=10) models trained in Notebook 05 on the `temp_next_{24,48,72}h` targets. All three models share the same feature set and scaler; the RMSE grows from ~3.4 °C at 24 h to ~5.1 °C at 72 h because weather decorrelates over time. `Rain 24h` is the aggregate calibrated probability from a 30-tree DecisionTree bagging ensemble, mapped through the Notebook 05 reliability curve. `Confidence` is ±1 sigma = ±RMSE on the held-out test. `Now (local)` is the city's local time at the reference row.

### Hourly Predictions (next 24 h, per city)

Click a city to expand. Each row is one hour; `temp` is our Ridge 24 h model rolled across the past 24 hours, `rain %` is the calibrated bagging probability, `precip` and `clouds` and `wind` come from Open-Meteo's own NWP forecast.

<details><summary><strong>🇧🇷 Sao Paulo</strong> — Cfa, America/Sao_Paulo</summary>

| +h | local time | temp | rain % | precip | clouds | wind | conditions |
|----|------------|------|--------|--------|--------|------|------------|
| +1 | 2026-04-15 00:00 | +19.6°C |  96% | 0.0 mm |  60% |  4.2 km/h | Cloudy |
| +2 | 2026-04-15 01:00 | +19.7°C |  96% | 0.0 mm |  63% |  3.8 km/h | Cloudy |
| +3 | 2026-04-15 02:00 | +19.3°C |  96% | 0.0 mm |  57% |  4.6 km/h | Cloudy |
| +4 | 2026-04-15 03:00 | +19.1°C |  96% | 0.0 mm |  58% |  2.9 km/h | Cloudy |
| +5 | 2026-04-15 04:00 | +18.9°C |  96% | 0.0 mm |  63% |  2.6 km/h | Cloudy |
| +6 | 2026-04-15 05:00 | +18.4°C |  96% | 0.0 mm | 100% |  2.5 km/h | Cloudy |
| +7 | 2026-04-15 06:00 | +18.0°C |  96% | 0.0 mm |  56% |  2.5 km/h | Cloudy |
| +8 | 2026-04-15 07:00 | +18.5°C |  96% | 0.0 mm |  20% |  2.3 km/h | Clear |
| +9 | 2026-04-15 08:00 | +19.4°C |  96% | 0.0 mm |  14% |  2.7 km/h | Clear |
| +10 | 2026-04-15 09:00 | +21.1°C |  96% | 0.0 mm |   7% |  5.8 km/h | Clear |
| +11 | 2026-04-15 10:00 | +22.9°C |  96% | 0.0 mm |  13% |  6.5 km/h | Clear |
| +12 | 2026-04-15 11:00 | +24.2°C |  96% | 0.0 mm |  28% |  7.7 km/h | Clear |
| +13 | 2026-04-15 12:00 | +25.3°C |  96% | 0.0 mm |  17% |  7.4 km/h | Clear |
| +14 | 2026-04-15 13:00 | +26.2°C |  96% | 0.0 mm |   0% |  6.8 km/h | Clear |
| +15 | 2026-04-15 14:00 | +26.4°C |  81% | 0.0 mm |  37% |  6.4 km/h | Clear |
| +16 | 2026-04-15 15:00 | +26.3°C |  81% | 0.0 mm |  50% |  7.4 km/h | Cloudy |
| +17 | 2026-04-15 16:00 | +25.8°C |  81% | 0.0 mm |  53% |  9.2 km/h | Cloudy |
| +18 | 2026-04-15 17:00 | +24.6°C |  96% | 0.0 mm |  41% | 10.9 km/h | Clear |
| +19 | 2026-04-15 18:00 | +22.7°C |  96% | 0.0 mm |  41% | 11.8 km/h | Clear |
| +20 | 2026-04-15 19:00 | +20.9°C |  96% | 0.0 mm |  32% | 10.0 km/h | Clear |
| +21 | 2026-04-15 20:00 | +19.8°C |  96% | 0.0 mm |  39% |  8.5 km/h | Clear |
| +22 | 2026-04-15 21:00 | +19.5°C |  96% | 0.0 mm |  28% |  7.4 km/h | Clear |
| +23 | 2026-04-15 22:00 | +19.2°C |  96% | 0.0 mm |  42% |  5.5 km/h | Clear |
| +24 | 2026-04-15 23:00 | +19.0°C |  96% | 0.0 mm |  44% |  4.4 km/h | Cloudy |

</details>

<details><summary><strong>🇧🇷 Rio de Janeiro</strong> — Aw, America/Sao_Paulo</summary>

| +h | local time | temp | rain % | precip | clouds | wind | conditions |
|----|------------|------|--------|--------|--------|------|------------|
| +1 | 2026-04-15 00:00 | +22.5°C |  81% | 0.0 mm |  65% |  2.9 km/h | Cloudy |
| +2 | 2026-04-15 01:00 | +21.6°C |  65% | 0.0 mm |  68% |  3.3 km/h | Cloudy |
| +3 | 2026-04-15 02:00 | +21.5°C |  65% | 0.0 mm |  57% |  2.9 km/h | Cloudy |
| +4 | 2026-04-15 03:00 | +21.4°C |  71% | 0.0 mm |  60% |  2.7 km/h | Cloudy |
| +5 | 2026-04-15 04:00 | +21.4°C |  71% | 0.0 mm |  67% |  2.7 km/h | Foggy |
| +6 | 2026-04-15 05:00 | +21.1°C |  71% | 0.0 mm |  76% |  3.3 km/h | Foggy |
| +7 | 2026-04-15 06:00 | +21.0°C |  65% | 0.0 mm |  75% |  3.2 km/h | Foggy |
| +8 | 2026-04-15 07:00 | +22.0°C |  81% | 0.0 mm |  59% |  3.7 km/h | Foggy |
| +9 | 2026-04-15 08:00 | +24.0°C |  81% | 0.0 mm |  49% |  3.6 km/h | Cloudy |
| +10 | 2026-04-15 09:00 | +26.1°C |  71% | 0.0 mm |  49% |  3.3 km/h | Cloudy |
| +11 | 2026-04-15 10:00 | +27.3°C |  65% | 0.0 mm |  63% |  2.3 km/h | Cloudy |
| +12 | 2026-04-15 11:00 | +28.2°C |  65% | 0.0 mm |  65% |  0.4 km/h | Cloudy |
| +13 | 2026-04-15 12:00 | +29.1°C |  62% | 0.0 mm |  81% |  2.1 km/h | Cloudy |
| +14 | 2026-04-15 13:00 | +29.4°C |  62% | 0.0 mm |  78% |  6.2 km/h | Cloudy |
| +15 | 2026-04-15 14:00 | +29.3°C |  71% | 0.0 mm |  90% |  8.6 km/h | Cloudy |
| +16 | 2026-04-15 15:00 | +28.9°C |  62% | 0.0 mm |  62% | 10.4 km/h | Cloudy |
| +17 | 2026-04-15 16:00 | +27.7°C |  71% | 0.0 mm |  10% |  9.3 km/h | Clear |
| +18 | 2026-04-15 17:00 | +26.6°C |  35% | 0.0 mm |   4% |  8.2 km/h | Clear |
| +19 | 2026-04-15 18:00 | +25.4°C |  35% | 0.0 mm |   0% |  6.6 km/h | Clear |
| +20 | 2026-04-15 19:00 | +24.5°C |  71% | 0.0 mm |   0% |  3.9 km/h | Clear |
| +21 | 2026-04-15 20:00 | +24.2°C |  65% | 0.0 mm |   0% |  2.3 km/h | Clear |
| +22 | 2026-04-15 21:00 | +23.9°C |  71% | 0.0 mm |   0% |  0.5 km/h | Clear |
| +23 | 2026-04-15 22:00 | +23.5°C |  65% | 0.0 mm |  10% |  1.8 km/h | Clear |
| +24 | 2026-04-15 23:00 | +22.8°C |  65% | 0.0 mm |  25% |  2.9 km/h | Clear |

</details>

<details><summary><strong>🇧🇷 Sao Jose dos Campos</strong> — Cfa, America/Sao_Paulo</summary>

| +h | local time | temp | rain % | precip | clouds | wind | conditions |
|----|------------|------|--------|--------|--------|------|------------|
| +1 | 2026-04-15 00:00 | +18.4°C |  81% | 0.0 mm |  52% |  1.1 km/h | Cloudy |
| +2 | 2026-04-15 01:00 | +18.2°C |  81% | 0.0 mm |  53% |  0.5 km/h | Cloudy |
| +3 | 2026-04-15 02:00 | +17.8°C |  65% | 0.0 mm |  53% |  1.3 km/h | Cloudy |
| +4 | 2026-04-15 03:00 | +18.0°C |  71% | 0.0 mm |  32% |  2.2 km/h | Cloudy |
| +5 | 2026-04-15 04:00 | +17.5°C |  71% | 0.0 mm |  45% |  3.3 km/h | Foggy |
| +6 | 2026-04-15 05:00 | +17.0°C |  65% | 0.0 mm | 100% |  3.5 km/h | Foggy |
| +7 | 2026-04-15 06:00 | +16.9°C |  65% | 0.0 mm |  89% |  2.9 km/h | Foggy |
| +8 | 2026-04-15 07:00 | +17.9°C |  65% | 0.0 mm |  63% |  2.6 km/h | Foggy |
| +9 | 2026-04-15 08:00 | +20.0°C |  71% | 0.0 mm |  63% |  1.3 km/h | Foggy |
| +10 | 2026-04-15 09:00 | +21.7°C |  71% | 0.0 mm |  38% |  4.1 km/h | Clear |
| +11 | 2026-04-15 10:00 | +23.0°C |  65% | 0.0 mm |  38% |  4.8 km/h | Clear |
| +12 | 2026-04-15 11:00 | +24.3°C |  71% | 0.0 mm |  30% |  5.2 km/h | Clear |
| +13 | 2026-04-15 12:00 | +25.5°C |  71% | 0.0 mm |   4% |  6.6 km/h | Clear |
| +14 | 2026-04-15 13:00 | +26.2°C |  71% | 0.0 mm |   0% |  7.4 km/h | Clear |
| +15 | 2026-04-15 14:00 | +26.5°C |  71% | 0.0 mm |   0% |  7.8 km/h | Clear |
| +16 | 2026-04-15 15:00 | +26.3°C |  71% | 0.0 mm |   0% |  7.6 km/h | Clear |
| +17 | 2026-04-15 16:00 | +25.4°C |  71% | 0.0 mm |   0% |  6.6 km/h | Clear |
| +18 | 2026-04-15 17:00 | +24.6°C |  65% | 0.0 mm |   6% |  4.5 km/h | Clear |
| +19 | 2026-04-15 18:00 | +23.1°C |  81% | 0.0 mm |  28% |  4.7 km/h | Clear |
| +20 | 2026-04-15 19:00 | +20.7°C |  65% | 0.0 mm |  40% |  5.4 km/h | Clear |
| +21 | 2026-04-15 20:00 | +20.1°C |  65% | 0.0 mm |  27% |  2.9 km/h | Clear |
| +22 | 2026-04-15 21:00 | +19.5°C |  65% | 0.0 mm |   0% |  2.8 km/h | Clear |
| +23 | 2026-04-15 22:00 | +19.0°C |  71% | 0.0 mm |  29% |  2.9 km/h | Clear |
| +24 | 2026-04-15 23:00 | +18.4°C |  65% | 0.0 mm |  42% |  2.3 km/h | Clear |

</details>

<details><summary><strong>🇧🇷 Campinas</strong> — Cfa, America/Sao_Paulo</summary>

| +h | local time | temp | rain % | precip | clouds | wind | conditions |
|----|------------|------|--------|--------|--------|------|------------|
| +1 | 2026-04-15 00:00 | +19.8°C |  96% | 0.0 mm |   4% | 14.9 km/h | Clear |
| +2 | 2026-04-15 01:00 | +19.3°C |  96% | 0.0 mm |  65% | 14.2 km/h | Cloudy |
| +3 | 2026-04-15 02:00 | +19.5°C |  96% | 0.0 mm | 100% | 12.3 km/h | Cloudy |
| +4 | 2026-04-15 03:00 | +19.0°C |  96% | 0.0 mm |  95% | 10.2 km/h | Cloudy |
| +5 | 2026-04-15 04:00 | +18.8°C |  96% | 0.0 mm |  29% |  9.1 km/h | Clear |
| +6 | 2026-04-15 05:00 | +18.3°C |  96% | 0.0 mm |  23% |  7.3 km/h | Clear |
| +7 | 2026-04-15 06:00 | +18.1°C |  96% | 0.0 mm |  14% |  6.4 km/h | Clear |
| +8 | 2026-04-15 07:00 | +18.7°C |  96% | 0.0 mm |  12% |  6.0 km/h | Clear |
| +9 | 2026-04-15 08:00 | +20.5°C |  96% | 0.0 mm |  13% |  8.8 km/h | Clear |
| +10 | 2026-04-15 09:00 | +22.6°C |  96% | 0.0 mm |   8% | 10.1 km/h | Clear |
| +11 | 2026-04-15 10:00 | +24.7°C |  96% | 0.0 mm |   0% |  7.8 km/h | Clear |
| +12 | 2026-04-15 11:00 | +25.4°C |  96% | 0.0 mm |   3% |  6.8 km/h | Clear |
| +13 | 2026-04-15 12:00 | +26.6°C |  96% | 0.0 mm |  27% |  2.9 km/h | Clear |
| +14 | 2026-04-15 13:00 | +27.6°C |  96% | 0.0 mm |  40% |  0.8 km/h | Clear |
| +15 | 2026-04-15 14:00 | +28.0°C |  96% | 0.0 mm |  54% |  2.5 km/h | Cloudy |
| +16 | 2026-04-15 15:00 | +27.9°C |  96% | 0.0 mm | 100% |  3.6 km/h | Cloudy |
| +17 | 2026-04-15 16:00 | +27.2°C |  96% | 0.0 mm |  63% |  5.5 km/h | Cloudy |
| +18 | 2026-04-15 17:00 | +26.4°C |  96% | 0.0 mm |  70% |  5.9 km/h | Cloudy |
| +19 | 2026-04-15 18:00 | +24.8°C |  96% | 0.0 mm |  65% |  4.3 km/h | Cloudy |
| +20 | 2026-04-15 19:00 | +23.2°C |  96% | 0.0 mm |  67% |  6.0 km/h | Cloudy |
| +21 | 2026-04-15 20:00 | +22.0°C |  96% | 0.0 mm |  63% |  8.1 km/h | Cloudy |
| +22 | 2026-04-15 21:00 | +20.9°C |  96% | 0.0 mm |  30% |  9.3 km/h | Clear |
| +23 | 2026-04-15 22:00 | +20.2°C |  96% | 0.0 mm |  28% | 10.0 km/h | Clear |
| +24 | 2026-04-15 23:00 | +19.6°C |  96% | 0.0 mm |  27% | 10.4 km/h | Clear |

</details>

<details><summary><strong>🇺🇸 New York</strong> — Dfa, America/New_York</summary>

| +h | local time | temp | rain % | precip | clouds | wind | conditions |
|----|------------|------|--------|--------|--------|------|------------|
| +1 | 2026-04-15 00:00 | +19.2°C |  81% | 0.0 mm |  99% | 10.9 km/h | Cloudy |
| +2 | 2026-04-15 01:00 | +18.1°C |  96% | 0.0 mm |  99% |  7.1 km/h | Cloudy |
| +3 | 2026-04-15 02:00 | +17.5°C |  81% | 0.0 mm |  99% |  6.4 km/h | Cloudy |
| +4 | 2026-04-15 03:00 | +16.7°C |  81% | 0.0 mm |  20% |  4.3 km/h | Clear |
| +5 | 2026-04-15 04:00 | +16.6°C |  81% | 0.0 mm |  81% |  6.2 km/h | Cloudy |
| +6 | 2026-04-15 05:00 | +15.8°C |  81% | 0.0 mm |   0% |  3.2 km/h | Clear |
| +7 | 2026-04-15 06:00 | +15.7°C |  81% | 0.0 mm |   9% |  4.5 km/h | Clear |
| +8 | 2026-04-15 07:00 | +16.0°C |  81% | 0.0 mm |   0% |  3.3 km/h | Clear |
| +9 | 2026-04-15 08:00 | +18.5°C |  81% | 0.0 mm |   0% |  9.2 km/h | Clear |
| +10 | 2026-04-15 09:00 | +21.2°C |  81% | 0.0 mm |  38% |  8.6 km/h | Clear |
| +11 | 2026-04-15 10:00 | +22.9°C |  81% | 0.0 mm |  88% | 11.0 km/h | Cloudy |
| +12 | 2026-04-15 11:00 | +24.9°C |  65% | 0.0 mm |  89% | 15.7 km/h | Cloudy |
| +13 | 2026-04-15 12:00 | +25.8°C |  65% | 0.0 mm | 100% | 17.1 km/h | Cloudy |
| +14 | 2026-04-15 13:00 | +27.2°C |  81% | 0.0 mm |   0% | 21.4 km/h | Clear |
| +15 | 2026-04-15 14:00 | +28.0°C |  65% | 0.0 mm |   0% | 21.9 km/h | Clear |
| +16 | 2026-04-15 15:00 | +27.6°C |  65% | 0.0 mm |  12% | 23.2 km/h | Clear |
| +17 | 2026-04-15 16:00 | +26.8°C |  81% | 0.0 mm | 100% | 25.7 km/h | Cloudy |
| +18 | 2026-04-15 17:00 | +26.3°C |  71% | 0.0 mm | 100% | 22.2 km/h | Cloudy |
| +19 | 2026-04-15 18:00 | +26.1°C |  71% | 0.0 mm | 100% | 19.6 km/h | Cloudy |
| +20 | 2026-04-15 19:00 | +25.6°C |  71% | 0.0 mm | 100% | 14.2 km/h | Cloudy |
| +21 | 2026-04-15 20:00 | +23.7°C |  65% | 0.0 mm |   0% |  8.3 km/h | Clear |
| +22 | 2026-04-15 21:00 | +21.4°C |  81% | 0.0 mm |  10% | 12.3 km/h | Clear |
| +23 | 2026-04-15 22:00 | +21.6°C |  96% | 0.0 mm | 100% | 11.8 km/h | Cloudy |
| +24 | 2026-04-15 23:00 | +21.2°C |  81% | 0.0 mm | 100% | 11.5 km/h | Cloudy |

</details>

<details><summary><strong>🇺🇸 Los Angeles</strong> — Csb, America/Los_Angeles</summary>

| +h | local time | temp | rain % | precip | clouds | wind | conditions |
|----|------------|------|--------|--------|--------|------|------------|
| +1 | 2026-04-15 00:00 | +15.0°C |  96% | 0.0 mm |   1% |  7.2 km/h | Clear |
| +2 | 2026-04-15 01:00 | +13.9°C |  96% | 0.0 mm |   8% |  7.2 km/h | Clear |
| +3 | 2026-04-15 02:00 | +13.2°C |  96% | 0.0 mm |  13% |  5.8 km/h | Clear |
| +4 | 2026-04-15 03:00 | +13.0°C |  96% | 0.0 mm |   1% |  5.4 km/h | Clear |
| +5 | 2026-04-15 04:00 | +12.9°C |  96% | 0.0 mm |  43% |  5.4 km/h | Clear |
| +6 | 2026-04-15 05:00 | +12.4°C |  96% | 0.0 mm |  85% |  5.1 km/h | Cloudy |
| +7 | 2026-04-15 06:00 | +12.6°C |  96% | 0.0 mm |  18% |  5.7 km/h | Clear |
| +8 | 2026-04-15 07:00 | +13.4°C |  96% | 0.0 mm |   7% |  5.7 km/h | Clear |
| +9 | 2026-04-15 08:00 | +15.5°C |  96% | 0.0 mm |   2% |  3.2 km/h | Clear |
| +10 | 2026-04-15 09:00 | +17.6°C |  96% | 0.0 mm |   1% |  4.7 km/h | Clear |
| +11 | 2026-04-15 10:00 | +19.2°C |  81% | 0.0 mm |   0% |  6.5 km/h | Clear |
| +12 | 2026-04-15 11:00 | +21.1°C |  81% | 0.0 mm |   0% |  6.9 km/h | Clear |
| +13 | 2026-04-15 12:00 | +22.2°C |  81% | 0.0 mm |   4% |  8.6 km/h | Clear |
| +14 | 2026-04-15 13:00 | +23.5°C |  81% | 0.0 mm |   3% | 15.0 km/h | Clear |
| +15 | 2026-04-15 14:00 | +24.2°C |  81% | 0.0 mm |   3% | 15.3 km/h | Clear |
| +16 | 2026-04-15 15:00 | +23.7°C |  81% | 0.0 mm |   3% | 15.3 km/h | Clear |
| +17 | 2026-04-15 16:00 | +23.2°C |  81% | 0.0 mm |   8% | 14.3 km/h | Clear |
| +18 | 2026-04-15 17:00 | +22.0°C |  81% | 0.0 mm |  84% | 13.6 km/h | Cloudy |
| +19 | 2026-04-15 18:00 | +20.9°C |  81% | 0.0 mm | 100% | 14.0 km/h | Cloudy |
| +20 | 2026-04-15 19:00 | +18.7°C |  81% | 0.0 mm | 100% |  9.2 km/h | Cloudy |
| +21 | 2026-04-15 20:00 | +17.7°C |  81% | 0.0 mm |  85% |  6.9 km/h | Cloudy |
| +22 | 2026-04-15 21:00 | +16.8°C |  96% | 0.0 mm |   7% |  4.2 km/h | Clear |
| +23 | 2026-04-15 22:00 | +16.3°C |  96% | 0.0 mm |  71% |  3.3 km/h | Cloudy |
| +24 | 2026-04-15 23:00 | +16.1°C |  96% | 0.0 mm | 100% |  5.3 km/h | Cloudy |

</details>

<details><summary><strong>🇬🇧 London</strong> — Cfb, Europe/London</summary>

| +h | local time | temp | rain % | precip | clouds | wind | conditions |
|----|------------|------|--------|--------|--------|------|------------|
| +1 | 2026-04-15 00:00 | +10.6°C |  96% | 0.0 mm | 100% | 11.9 km/h | Cloudy |
| +2 | 2026-04-15 01:00 | +10.0°C |  96% | 0.0 mm | 100% | 11.5 km/h | Cloudy |
| +3 | 2026-04-15 02:00 | +9.4°C |  96% | 0.0 mm | 100% | 12.6 km/h | Cloudy |
| +4 | 2026-04-15 03:00 | +9.4°C |  96% | 0.0 mm | 100% | 13.0 km/h | Cloudy |
| +5 | 2026-04-15 04:00 | +9.1°C |  96% | 0.0 mm | 100% | 13.0 km/h | Cloudy |
| +6 | 2026-04-15 05:00 | +8.4°C |  96% | 0.0 mm | 100% | 13.7 km/h | Cloudy |
| +7 | 2026-04-15 06:00 | +8.5°C |  96% | 0.0 mm | 100% | 14.0 km/h | Cloudy |
| +8 | 2026-04-15 07:00 | +8.9°C |  96% | 0.0 mm | 100% | 13.0 km/h | Cloudy |
| +9 | 2026-04-15 08:00 | +10.4°C |  96% | 0.0 mm | 100% | 14.8 km/h | Cloudy |
| +10 | 2026-04-15 09:00 | +12.1°C |  96% | 0.0 mm | 100% | 13.7 km/h | Cloudy |
| +11 | 2026-04-15 10:00 | +13.3°C |  96% | 0.0 mm |  87% | 11.9 km/h | Cloudy |
| +12 | 2026-04-15 11:00 | +14.9°C |  96% | 0.0 mm | 100% | 15.8 km/h | Cloudy |
| +13 | 2026-04-15 12:00 | +15.5°C |  96% | 0.0 mm | 100% | 18.7 km/h | Cloudy |
| +14 | 2026-04-15 13:00 | +16.2°C |  96% | 0.1 mm |  99% | 16.6 km/h | Rainy |
| +15 | 2026-04-15 14:00 | +16.7°C |  96% | 0.0 mm |  88% | 17.3 km/h | Cloudy |
| +16 | 2026-04-15 15:00 | +17.0°C |  96% | 0.0 mm |  88% | 20.2 km/h | Cloudy |
| +17 | 2026-04-15 16:00 | +16.8°C |  96% | 0.0 mm |  83% | 23.8 km/h | Cloudy |
| +18 | 2026-04-15 17:00 | +16.1°C |  96% | 1.4 mm | 100% | 20.9 km/h | Rainy |
| +19 | 2026-04-15 18:00 | +15.3°C |  96% | 0.0 mm |  36% | 17.6 km/h | Clear |
| +20 | 2026-04-15 19:00 | +15.1°C |  96% | 0.0 mm |  35% | 17.6 km/h | Clear |
| +21 | 2026-04-15 20:00 | +14.9°C |  96% | 0.0 mm |  30% | 18.7 km/h | Clear |
| +22 | 2026-04-15 21:00 | +14.8°C |  96% | 0.0 mm |  23% | 19.1 km/h | Clear |
| +23 | 2026-04-15 22:00 | +14.7°C |  96% | 0.0 mm |   0% | 17.3 km/h | Clear |
| +24 | 2026-04-15 23:00 | +14.4°C |  71% | 0.0 mm |   9% | 14.8 km/h | Clear |

</details>

<details><summary><strong>🇩🇪 Berlin</strong> — Cfb, Europe/Berlin</summary>

| +h | local time | temp | rain % | precip | clouds | wind | conditions |
|----|------------|------|--------|--------|--------|------|------------|
| +1 | 2026-04-15 00:00 | +11.7°C |  65% | 0.0 mm |   0% |  6.4 km/h | Clear |
| +2 | 2026-04-15 01:00 | +11.2°C |  65% | 0.0 mm |   0% |  5.6 km/h | Clear |
| +3 | 2026-04-15 02:00 | +11.2°C |  65% | 0.0 mm |   0% |  3.9 km/h | Clear |
| +4 | 2026-04-15 03:00 | +10.5°C |  71% | 0.0 mm |   0% |  3.6 km/h | Clear |
| +5 | 2026-04-15 04:00 | +10.0°C |  71% | 0.0 mm |   0% |  3.8 km/h | Clear |
| +6 | 2026-04-15 05:00 | +9.7°C |  71% | 0.0 mm |  37% |  1.8 km/h | Clear |
| +7 | 2026-04-15 06:00 | +9.7°C |  71% | 0.0 mm |  20% |  2.3 km/h | Foggy |
| +8 | 2026-04-15 07:00 | +9.8°C |  71% | 0.0 mm |   0% |  1.1 km/h | Clear |
| +9 | 2026-04-15 08:00 | +10.2°C |  96% | 0.0 mm |   0% |  1.8 km/h | Clear |
| +10 | 2026-04-15 09:00 | +10.7°C |  96% | 0.0 mm |   0% |  1.1 km/h | Clear |
| +11 | 2026-04-15 10:00 | +11.5°C |  96% | 0.0 mm |   0% |  1.0 km/h | Clear |
| +12 | 2026-04-15 11:00 | +12.1°C |  96% | 0.0 mm |   0% |  2.9 km/h | Clear |
| +13 | 2026-04-15 12:00 | +12.7°C |  96% | 0.0 mm |   7% |  1.5 km/h | Clear |
| +14 | 2026-04-15 13:00 | +13.2°C |  96% | 0.0 mm | 100% |  1.0 km/h | Cloudy |
| +15 | 2026-04-15 14:00 | +13.6°C |  96% | 0.0 mm |  95% |  1.8 km/h | Clear |
| +16 | 2026-04-15 15:00 | +14.0°C |  96% | 0.0 mm |  53% |  1.6 km/h | Cloudy |
| +17 | 2026-04-15 16:00 | +14.1°C |  96% | 0.0 mm |  43% |  6.3 km/h | Clear |
| +18 | 2026-04-15 17:00 | +13.2°C |  96% | 0.0 mm |  48% |  5.5 km/h | Clear |
| +19 | 2026-04-15 18:00 | +12.9°C |  96% | 0.0 mm |  88% |  6.6 km/h | Cloudy |
| +20 | 2026-04-15 19:00 | +12.5°C |  96% | 0.0 mm | 100% |  4.9 km/h | Cloudy |
| +21 | 2026-04-15 20:00 | +12.6°C |  96% | 0.0 mm | 100% |  6.1 km/h | Cloudy |
| +22 | 2026-04-15 21:00 | +12.2°C |  96% | 0.0 mm |  97% |  8.0 km/h | Cloudy |
| +23 | 2026-04-15 22:00 | +12.2°C |  96% | 0.0 mm |  84% |  6.2 km/h | Cloudy |
| +24 | 2026-04-15 23:00 | +12.0°C |  96% | 0.0 mm |  85% |  5.9 km/h | Cloudy |

</details>

<details><summary><strong>🇳🇴 Oslo</strong> — Dfb, Europe/Oslo</summary>

| +h | local time | temp | rain % | precip | clouds | wind | conditions |
|----|------------|------|--------|--------|--------|------|------------|
| +1 | 2026-04-15 00:00 | +10.2°C |  96% | 0.6 mm | 100% |  6.8 km/h | Rainy |
| +2 | 2026-04-15 01:00 | +9.5°C |  96% | 0.2 mm | 100% | 11.2 km/h | Rainy |
| +3 | 2026-04-15 02:00 | +8.5°C |  81% | 0.2 mm | 100% | 13.7 km/h | Rainy |
| +4 | 2026-04-15 03:00 | +7.9°C |  81% | 0.0 mm | 100% | 16.6 km/h | Cloudy |
| +5 | 2026-04-15 04:00 | +7.4°C |  81% | 0.0 mm | 100% | 19.1 km/h | Cloudy |
| +6 | 2026-04-15 05:00 | +6.7°C |  81% | 0.1 mm | 100% | 19.4 km/h | Rainy |
| +7 | 2026-04-15 06:00 | +6.1°C |  81% | 0.0 mm | 100% | 19.1 km/h | Cloudy |
| +8 | 2026-04-15 07:00 | +6.2°C |  81% | 0.0 mm |  99% | 21.2 km/h | Cloudy |
| +9 | 2026-04-15 08:00 | +6.5°C |  81% | 0.0 mm |  99% | 20.9 km/h | Cloudy |
| +10 | 2026-04-15 09:00 | +7.1°C |  81% | 0.0 mm |  96% | 19.1 km/h | Cloudy |
| +11 | 2026-04-15 10:00 | +8.2°C |  81% | 0.0 mm |  99% | 18.4 km/h | Cloudy |
| +12 | 2026-04-15 11:00 | +9.4°C |  65% | 0.0 mm | 100% | 15.5 km/h | Cloudy |
| +13 | 2026-04-15 12:00 | +10.4°C |  81% | 0.0 mm |  76% | 14.8 km/h | Cloudy |
| +14 | 2026-04-15 13:00 | +11.2°C |  81% | 0.0 mm |  85% | 14.0 km/h | Cloudy |
| +15 | 2026-04-15 14:00 | +11.8°C |  81% | 0.0 mm |  71% | 10.8 km/h | Cloudy |
| +16 | 2026-04-15 15:00 | +12.1°C |  96% | 0.0 mm |  70% | 10.4 km/h | Cloudy |
| +17 | 2026-04-15 16:00 | +12.2°C |  96% | 0.0 mm |  75% | 11.5 km/h | Cloudy |
| +18 | 2026-04-15 17:00 | +12.5°C |  96% | 0.0 mm |  75% | 12.6 km/h | Cloudy |
| +19 | 2026-04-15 18:00 | +12.5°C |  81% | 0.0 mm |  78% | 13.7 km/h | Cloudy |
| +20 | 2026-04-15 19:00 | +11.7°C |  81% | 0.0 mm |  66% | 11.9 km/h | Cloudy |
| +21 | 2026-04-15 20:00 | +11.1°C |  96% | 0.0 mm |  79% |  9.0 km/h | Cloudy |
| +22 | 2026-04-15 21:00 | +10.7°C |  96% | 0.0 mm |  49% | 10.4 km/h | Clear |
| +23 | 2026-04-15 22:00 | +9.7°C |  96% | 0.0 mm |  72% | 10.4 km/h | Cloudy |
| +24 | 2026-04-15 23:00 | +9.6°C |  81% | 0.0 mm |  98% | 11.2 km/h | Cloudy |

</details>

<details><summary><strong>🇯🇵 Tokyo</strong> — Cfa, Asia/Tokyo</summary>

| +h | local time | temp | rain % | precip | clouds | wind | conditions |
|----|------------|------|--------|--------|--------|------|------------|
| +1 | 2026-04-16 00:00 | +15.2°C |  96% | 2.5 mm |  84% |  6.5 km/h | Rainy |
| +2 | 2026-04-16 01:00 | +14.3°C |  96% | 1.8 mm |  69% |  6.8 km/h | Rainy |
| +3 | 2026-04-16 02:00 | +14.1°C |  96% | 1.1 mm |  86% |  7.1 km/h | Rainy |
| +4 | 2026-04-16 03:00 | +13.8°C |  96% | 0.3 mm |  79% |  7.4 km/h | Rainy |
| +5 | 2026-04-16 04:00 | +13.5°C |  96% | 0.8 mm |  84% |  7.7 km/h | Rainy |
| +6 | 2026-04-16 05:00 | +13.1°C |  96% | 0.1 mm |  87% |  7.5 km/h | Rainy |
| +7 | 2026-04-16 06:00 | +13.6°C |  81% | 0.0 mm |  99% |  9.0 km/h | Cloudy |
| +8 | 2026-04-16 07:00 | +13.8°C |  81% | 0.0 mm |  93% |  8.7 km/h | Cloudy |
| +9 | 2026-04-16 08:00 | +14.5°C |  81% | 0.0 mm |  54% |  9.0 km/h | Cloudy |
| +10 | 2026-04-16 09:00 | +15.4°C |  81% | 0.0 mm |  45% |  9.5 km/h | Clear |
| +11 | 2026-04-16 10:00 | +16.3°C |  81% | 0.0 mm |  34% | 10.1 km/h | Clear |
| +12 | 2026-04-16 11:00 | +17.8°C |  81% | 0.0 mm |   7% | 10.4 km/h | Clear |
| +13 | 2026-04-16 12:00 | +19.0°C |  81% | 0.0 mm |  12% | 10.1 km/h | Clear |
| +14 | 2026-04-16 13:00 | +19.9°C |  81% | 0.0 mm |  12% | 10.9 km/h | Clear |
| +15 | 2026-04-16 14:00 | +20.3°C |  96% | 0.0 mm |  16% | 10.2 km/h | Clear |
| +16 | 2026-04-16 15:00 | +20.2°C |  96% | 0.0 mm |  40% |  8.5 km/h | Clear |
| +17 | 2026-04-16 16:00 | +19.7°C |  96% | 0.0 mm |  46% |  8.7 km/h | Clear |
| +18 | 2026-04-16 17:00 | +18.7°C |  96% | 0.0 mm |  50% |  8.4 km/h | Cloudy |
| +19 | 2026-04-16 18:00 | +17.2°C |  96% | 0.0 mm |  40% |  8.2 km/h | Clear |
| +20 | 2026-04-16 19:00 | +16.3°C |  96% | 0.0 mm |  44% |  7.4 km/h | Clear |
| +21 | 2026-04-16 20:00 | +15.7°C |  96% | 0.0 mm |  34% |  6.7 km/h | Clear |
| +22 | 2026-04-16 21:00 | +15.2°C |  96% | 0.0 mm |  33% |  6.4 km/h | Clear |
| +23 | 2026-04-16 22:00 | +13.4°C |  96% | 0.0 mm |  24% |  5.6 km/h | Clear |
| +24 | 2026-04-16 23:00 | +11.9°C |  96% | 0.0 mm |  22% |  5.1 km/h | Clear |

</details>

<details><summary><strong>🇨🇳 Shanghai</strong> — Cfa, Asia/Shanghai</summary>

| +h | local time | temp | rain % | precip | clouds | wind | conditions |
|----|------------|------|--------|--------|--------|------|------------|
| +1 | 2026-04-15 00:00 | +15.3°C |  96% | 0.0 mm |  98% |  3.3 km/h | Cloudy |
| +2 | 2026-04-15 01:00 | +15.0°C |  96% | 0.0 mm |  99% |  2.9 km/h | Foggy |
| +3 | 2026-04-15 02:00 | +14.8°C |  96% | 0.0 mm |  98% |  4.0 km/h | Cloudy |
| +4 | 2026-04-15 03:00 | +14.4°C |  96% | 0.0 mm |  99% |  4.0 km/h | Cloudy |
| +5 | 2026-04-15 04:00 | +14.4°C |  96% | 0.0 mm |  98% |  4.7 km/h | Cloudy |
| +6 | 2026-04-15 05:00 | +14.5°C |  96% | 0.0 mm |  96% |  4.7 km/h | Cloudy |
| +7 | 2026-04-15 06:00 | +14.6°C |  96% | 0.0 mm |  96% |  5.1 km/h | Cloudy |
| +8 | 2026-04-15 07:00 | +15.4°C |  96% | 0.0 mm |  90% |  5.0 km/h | Cloudy |
| +9 | 2026-04-15 08:00 | +16.2°C |  96% | 0.0 mm |  81% |  6.8 km/h | Cloudy |
| +10 | 2026-04-15 09:00 | +16.5°C |  96% | 0.0 mm |  59% |  7.3 km/h | Cloudy |
| +11 | 2026-04-15 10:00 | +17.1°C |  96% | 0.0 mm |  60% |  6.7 km/h | Cloudy |
| +12 | 2026-04-15 11:00 | +17.7°C |  96% | 0.0 mm |  47% |  7.0 km/h | Cloudy |
| +13 | 2026-04-15 12:00 | +17.9°C |  96% | 0.0 mm |  55% |  8.0 km/h | Cloudy |
| +14 | 2026-04-15 13:00 | +18.2°C |  96% | 0.0 mm |  90% |  9.9 km/h | Cloudy |
| +15 | 2026-04-15 14:00 | +18.7°C |  96% | 0.0 mm | 100% | 10.4 km/h | Cloudy |
| +16 | 2026-04-15 15:00 | +18.6°C |  96% | 0.0 mm | 100% | 10.5 km/h | Cloudy |
| +17 | 2026-04-15 16:00 | +18.3°C |  96% | 0.0 mm | 100% | 10.9 km/h | Cloudy |
| +18 | 2026-04-15 17:00 | +17.8°C |  96% | 0.0 mm | 100% | 10.5 km/h | Cloudy |
| +19 | 2026-04-15 18:00 | +17.2°C |  96% | 0.0 mm |  92% |  8.8 km/h | Cloudy |
| +20 | 2026-04-15 19:00 | +16.7°C |  96% | 0.0 mm | 100% |  7.6 km/h | Cloudy |
| +21 | 2026-04-15 20:00 | +16.5°C |  96% | 0.0 mm | 100% |  6.5 km/h | Cloudy |
| +22 | 2026-04-15 21:00 | +16.3°C |  96% | 0.0 mm |  76% |  6.2 km/h | Cloudy |
| +23 | 2026-04-15 22:00 | +15.8°C |  96% | 0.0 mm |  97% |  5.2 km/h | Cloudy |
| +24 | 2026-04-15 23:00 | +15.8°C |  96% | 0.0 mm |  67% |  4.7 km/h | Cloudy |

</details>

<details><summary><strong>🇨🇳 Chongqing</strong> — Cwa, Asia/Shanghai</summary>

| +h | local time | temp | rain % | precip | clouds | wind | conditions |
|----|------------|------|--------|--------|--------|------|------------|
| +1 | 2026-04-15 00:00 | +18.6°C |  96% | 0.1 mm | 100% |  3.7 km/h | Cloudy |
| +2 | 2026-04-15 01:00 | +18.0°C |  96% | 0.4 mm | 100% |  5.1 km/h | Rainy |
| +3 | 2026-04-15 02:00 | +18.0°C |  96% | 0.1 mm |  99% |  1.3 km/h | Foggy |
| +4 | 2026-04-15 03:00 | +18.0°C |  96% | 0.2 mm |  93% |  3.7 km/h | Rainy |
| +5 | 2026-04-15 04:00 | +18.1°C |  96% | 0.1 mm |  98% |  4.0 km/h | Foggy |
| +6 | 2026-04-15 05:00 | +18.1°C |  96% | 0.0 mm | 100% |  3.6 km/h | Foggy |
| +7 | 2026-04-15 06:00 | +18.6°C |  96% | 0.2 mm | 100% |  3.1 km/h | Foggy |
| +8 | 2026-04-15 07:00 | +19.0°C |  96% | 0.0 mm | 100% |  3.9 km/h | Foggy |
| +9 | 2026-04-15 08:00 | +19.4°C |  96% | 0.0 mm |  99% |  3.5 km/h | Foggy |
| +10 | 2026-04-15 09:00 | +20.3°C |  96% | 0.0 mm | 100% |  1.3 km/h | Cloudy |
| +11 | 2026-04-15 10:00 | +21.2°C |  96% | 0.0 mm | 100% |  2.0 km/h | Foggy |
| +12 | 2026-04-15 11:00 | +22.0°C |  96% | 0.0 mm | 100% |  3.4 km/h | Cloudy |
| +13 | 2026-04-15 12:00 | +22.4°C |  96% | 0.0 mm |  97% |  3.2 km/h | Cloudy |
| +14 | 2026-04-15 13:00 | +22.8°C |  96% | 0.0 mm |  99% |  3.8 km/h | Cloudy |
| +15 | 2026-04-15 14:00 | +23.1°C |  96% | 0.0 mm |  99% |  8.2 km/h | Cloudy |
| +16 | 2026-04-15 15:00 | +22.9°C |  96% | 0.0 mm |  98% |  7.4 km/h | Cloudy |
| +17 | 2026-04-15 16:00 | +22.8°C |  96% | 0.0 mm |  80% | 12.2 km/h | Cloudy |
| +18 | 2026-04-15 17:00 | +23.0°C |  96% | 0.0 mm |  55% | 14.9 km/h | Cloudy |
| +19 | 2026-04-15 18:00 | +22.6°C |  96% | 0.0 mm |  55% | 12.4 km/h | Cloudy |
| +20 | 2026-04-15 19:00 | +21.8°C |  96% | 0.0 mm |  37% |  9.4 km/h | Clear |
| +21 | 2026-04-15 20:00 | +20.8°C |  96% | 0.0 mm |  39% |  7.1 km/h | Clear |
| +22 | 2026-04-15 21:00 | +19.6°C |  96% | 0.0 mm |  87% |  5.9 km/h | Cloudy |
| +23 | 2026-04-15 22:00 | +18.9°C |  96% | 0.0 mm |  90% |  5.6 km/h | Cloudy |
| +24 | 2026-04-15 23:00 | +18.5°C |  96% | 0.0 mm |  89% |  6.4 km/h | Cloudy |

</details>

<details><summary><strong>🇨🇳 Nanjing</strong> — Cfa, Asia/Shanghai</summary>

| +h | local time | temp | rain % | precip | clouds | wind | conditions |
|----|------------|------|--------|--------|--------|------|------------|
| +1 | 2026-04-15 00:00 | +14.3°C |  96% | 0.0 mm |  98% |  2.6 km/h | Cloudy |
| +2 | 2026-04-15 01:00 | +14.2°C |  96% | 0.0 mm |  97% |  2.9 km/h | Cloudy |
| +3 | 2026-04-15 02:00 | +13.9°C |  96% | 0.0 mm |  93% |  2.1 km/h | Cloudy |
| +4 | 2026-04-15 03:00 | +13.9°C |  96% | 0.0 mm |  90% |  2.1 km/h | Cloudy |
| +5 | 2026-04-15 04:00 | +14.0°C |  96% | 0.0 mm |  93% |  2.6 km/h | Foggy |
| +6 | 2026-04-15 05:00 | +13.7°C |  96% | 0.0 mm |  94% |  2.6 km/h | Foggy |
| +7 | 2026-04-15 06:00 | +13.9°C |  96% | 0.0 mm |  89% |  2.6 km/h | Cloudy |
| +8 | 2026-04-15 07:00 | +14.2°C |  96% | 0.0 mm |  62% |  2.1 km/h | Cloudy |
| +9 | 2026-04-15 08:00 | +15.1°C |  96% | 0.0 mm |  53% |  3.3 km/h | Clear |
| +10 | 2026-04-15 09:00 | +16.2°C |  96% | 0.0 mm |  34% |  5.2 km/h | Clear |
| +11 | 2026-04-15 10:00 | +17.3°C |  96% | 0.0 mm |  33% |  6.6 km/h | Clear |
| +12 | 2026-04-15 11:00 | +18.6°C |  96% | 0.0 mm |  51% |  6.5 km/h | Cloudy |
| +13 | 2026-04-15 12:00 | +19.4°C |  96% | 0.0 mm |  95% |  7.6 km/h | Cloudy |
| +14 | 2026-04-15 13:00 | +19.8°C |  96% | 0.0 mm | 100% |  8.8 km/h | Cloudy |
| +15 | 2026-04-15 14:00 | +20.0°C |  96% | 0.0 mm | 100% |  9.8 km/h | Cloudy |
| +16 | 2026-04-15 15:00 | +19.8°C |  96% | 0.0 mm | 100% | 11.9 km/h | Cloudy |
| +17 | 2026-04-15 16:00 | +19.3°C |  96% | 0.0 mm | 100% | 12.6 km/h | Cloudy |
| +18 | 2026-04-15 17:00 | +18.8°C |  96% | 0.0 mm | 100% | 11.2 km/h | Cloudy |
| +19 | 2026-04-15 18:00 | +18.3°C |  96% | 0.0 mm |  82% |  9.5 km/h | Cloudy |
| +20 | 2026-04-15 19:00 | +17.6°C |  96% | 0.0 mm |  94% |  6.6 km/h | Cloudy |
| +21 | 2026-04-15 20:00 | +17.2°C |  96% | 0.0 mm |  58% |  6.7 km/h | Cloudy |
| +22 | 2026-04-15 21:00 | +16.9°C |  96% | 0.0 mm |  99% |  5.6 km/h | Cloudy |
| +23 | 2026-04-15 22:00 | +16.7°C |  96% | 0.0 mm | 100% |  7.6 km/h | Cloudy |
| +24 | 2026-04-15 23:00 | +16.5°C |  96% | 0.0 mm |  91% |  8.3 km/h | Cloudy |

</details>

<details><summary><strong>🇦🇪 Dubai</strong> — BWh, Asia/Dubai</summary>

| +h | local time | temp | rain % | precip | clouds | wind | conditions |
|----|------------|------|--------|--------|--------|------|------------|
| +1 | 2026-04-15 00:00 | +24.9°C |  96% | 0.0 mm |  12% | 12.4 km/h | Clear |
| +2 | 2026-04-15 01:00 | +25.1°C |  96% | 0.0 mm |   0% | 12.5 km/h | Clear |
| +3 | 2026-04-15 02:00 | +25.0°C |  96% | 0.0 mm |   0% | 12.5 km/h | Clear |
| +4 | 2026-04-15 03:00 | +24.5°C |  96% | 0.0 mm |   0% | 11.8 km/h | Clear |
| +5 | 2026-04-15 04:00 | +25.1°C |  96% | 0.0 mm |   8% | 12.0 km/h | Clear |
| +6 | 2026-04-15 05:00 | +24.9°C |  96% | 0.0 mm |   7% | 12.2 km/h | Clear |
| +7 | 2026-04-15 06:00 | +25.2°C |  96% | 0.0 mm |  29% | 12.0 km/h | Clear |
| +8 | 2026-04-15 07:00 | +25.6°C |  96% | 0.0 mm |  16% | 12.5 km/h | Clear |
| +9 | 2026-04-15 08:00 | +26.3°C |  96% | 0.0 mm |   3% | 13.5 km/h | Clear |
| +10 | 2026-04-15 09:00 | +26.6°C |  96% | 0.0 mm |  35% | 14.8 km/h | Clear |
| +11 | 2026-04-15 10:00 | +27.1°C |  96% | 0.0 mm |  71% | 14.9 km/h | Cloudy |
| +12 | 2026-04-15 11:00 | +28.0°C |  96% | 0.0 mm |  70% | 17.6 km/h | Cloudy |
| +13 | 2026-04-15 12:00 | +28.4°C |  96% | 0.0 mm |  62% | 19.5 km/h | Cloudy |
| +14 | 2026-04-15 13:00 | +28.2°C |  96% | 0.0 mm |  74% | 22.2 km/h | Cloudy |
| +15 | 2026-04-15 14:00 | +28.1°C |  96% | 0.0 mm |  55% | 24.0 km/h | Cloudy |
| +16 | 2026-04-15 15:00 | +27.8°C |  96% | 0.0 mm |  78% | 24.9 km/h | Cloudy |
| +17 | 2026-04-15 16:00 | +27.7°C |  96% | 0.0 mm |  52% | 24.1 km/h | Cloudy |
| +18 | 2026-04-15 17:00 | +26.8°C |  96% | 0.0 mm |  70% | 21.8 km/h | Cloudy |
| +19 | 2026-04-15 18:00 | +26.4°C |  96% | 0.0 mm | 100% | 20.5 km/h | Cloudy |
| +20 | 2026-04-15 19:00 | +25.6°C |  96% | 0.0 mm | 100% | 21.2 km/h | Cloudy |
| +21 | 2026-04-15 20:00 | +25.6°C |  96% | 0.0 mm | 100% | 16.6 km/h | Cloudy |
| +22 | 2026-04-15 21:00 | +25.4°C |  96% | 0.0 mm | 100% | 17.8 km/h | Cloudy |
| +23 | 2026-04-15 22:00 | +25.0°C |  96% | 0.0 mm | 100% | 20.5 km/h | Cloudy |
| +24 | 2026-04-15 23:00 | +24.9°C |  96% | 0.0 mm | 100% | 21.2 km/h | Cloudy |

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
| Successful cities | 14 / 14 |


### Model Performance (held-out test set)

| Metric | Value |
|--------|-------|
| Model | Ridge (alpha=10) |
| Test RMSE (24 h temperature) | 3.41 °C |
| 95 % CI of RMSE | [3.32, 3.50] °C |
| Skill vs persistence-24 h | +0.264 |
| Post-hoc bias correction  | +0.68 °C |
| Successful cities         | 14 / 14 |


### Model Performance (held-out test set)

| Metric | Value |
|--------|-------|
| Model | Ridge (alpha=10) |
| Test RMSE (24 h temperature) | 3.41 °C |
| 95 % CI of RMSE | [3.32, 3.50] °C |
| Skill vs persistence-24 h | +0.264 |
| Post-hoc bias correction  | +0.68 °C |
| Successful cities         | 14 / 14 |


### Model Performance (Last 7 Days)

| Metric | Rain Prediction | Condition | Temp 24h | Temp 48h | Temp 72h |
|--------|-----------------|-----------|----------|----------|----------|
| Accuracy/RMSE | --% | --% | --°C | --°C | --°C |
| vs Baseline | -- | -- | -- | -- | -- |

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

| Region | Cities |
|--------|--------|
| Brazil 🇧🇷 | São Paulo, Rio de Janeiro, São José dos Campos, Campinas |
| USA 🇺🇸 | New York, Los Angeles |
| Europe 🇬🇧🇩🇪🇳🇴 | London, Berlin, Oslo |
| Asia 🇯🇵🇨🇳🇦🇪 | Tokyo, Shanghai, Chongqing, Nanjing, Dubai |

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
git clone https://github.com/yourusername/RustForMachineLearning.git
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

| Model | Rain (Acc) | Rain (F1) | Condition (Acc) | Condition (F1) |
|-------|------------|-----------|-----------------|----------------|
| Logistic Regression | -- | -- | -- | -- |
| Decision Tree | -- | -- | -- | -- |
| Random Forest | -- | -- | -- | -- |
| Gradient Boosting | -- | -- | -- | -- |
| Neural Network | -- | -- | -- | -- |
| **Ensemble** | -- | -- | -- | -- |

### Regression Tasks

| Model | Temp 24h (RMSE) | Temp 48h (RMSE) | Temp 72h (RMSE) |
|-------|-----------------|-----------------|-----------------|
| Linear Regression | -- | -- | -- |
| Decision Tree | -- | -- | -- |
| Random Forest | -- | -- | -- |
| Gradient Boosting | -- | -- | -- |
| Neural Network | -- | -- | -- |
| **Ensemble** | -- | -- | -- |

---

## 🔧 Technologies

### ML Libraries

| Library | Purpose | Status |
|---------|---------|--------|
| [linfa](https://github.com/rust-ml/linfa) | Classical ML | ✅ |
| [smartcore](https://github.com/smartcorelib/smartcore) | Classical ML | ✅ |
| [rustyml](https://github.com/rustyml/rustyml) | Classical ML | 🔄 |
| [Burn](https://github.com/tracel-ai/burn) | Deep Learning | 🔄 |

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
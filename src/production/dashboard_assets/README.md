# RustWeatherML — Dashboard

Two dashboards are shipped in this folder:

| File | Purpose |
|------|---------|
| `index.html` | Static, self-contained HTML (Chart.js from CDN) that reads the JSON files in `data/predictions/` and `data/monitoring_history/`. Open with any browser or serve with GitHub Pages. |
| `grafana_dashboard.json` | Importable Grafana dashboard definition. Requires the [Infinity datasource plugin](https://grafana.com/grafana/plugins/yesoreyeram-infinity-datasource/) pointed at the same JSON files. |

## Option A — GitHub Pages (static HTML)

1. Commit `docs/dashboard/index.html` (the daily_predictions binary writes it).
2. Enable GitHub Pages in repository settings pointing at the `docs/` folder on the `main` branch.
3. Navigate to `https://<user>.github.io/<repo>/dashboard/`.

The page fetches the last 14 days of prediction JSONs relative to the `docs/dashboard/` directory. No backend required.

## Option B — Grafana (local or cloud)

1. Install the **Infinity** datasource plugin (`yesoreyeram-infinity-datasource`).
2. Create an Infinity datasource named `rwml-infinity`, base URL = raw GitHub content URL or a local file server pointing at the repo root.
3. Import `grafana_dashboard.json` via **Dashboards → Import → Upload JSON file**.
4. Grafana will pick up `data/predictions/latest.json` and `data/monitoring_history/all.json`. If you prefer per-day files, tweak the URL field in each panel.

## Data contract

Both dashboards consume the same JSON shape described in
`src/production/predict.rs` (`DailyReport`) and
`src/production/drift_monitor.rs` (`DriftSnapshot`). Any change to those
schemas requires both dashboards to be refreshed.

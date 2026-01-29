# BeautyPulse 💄📊
A data engineering + analytics project using Open Beauty Facts.

## What it does
- Ingests real cosmetics data from Open Beauty Facts (skincare + makeup)
- Loads to Postgres (Neon)
- Stores daily snapshots + pipeline run logs
- Streamlit dashboard with:
  - KPIs + data quality
  - Product Explorer (Table + Cards with images/links)
  - Ingredients Analytics (top ingredients + alcohol/fragrance/paraben flags)
  - Brand insights + pipeline status

## Tech
Python, Requests, Pandas, SQLAlchemy, Postgres (Neon), Streamlit

## How to run
1. Create `.env`:
   DATABASE_URL=...
2. Run pipeline:
   python ingest/fetch_and_load.py
3. Run dashboard:
   streamlit run app/streamlit_app.py

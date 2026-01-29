import os
import uuid
import datetime as dt
import requests
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env file")

# Open Beauty Facts (cosmetics-only database)
OBF_BASE = "https://world.openbeautyfacts.org"

# 2 streams: skincare + makeup (keyword-based search queries)
SEARCH_STREAMS = [
    ("skincare", ["moisturizer", "cleanser", "serum", "sunscreen", "toner", "lotion", "cream"]),
    ("makeup", ["lipstick", "mascara", "foundation", "concealer", "eyeliner", "blush", "palette", "primer"]),
]

# Fields to request from OBF
OBF_FIELDS = [
    "code",
    "product_name",
    "brands",
    "categories",
    "categories_tags",
    "image_front_url",
    "url",
    "last_modified_t",
    "ingredients_text",
]

def obf_search(search_terms: str, page: int = 1, page_size: int = 50) -> dict:
    """
    OFF-style search endpoint supported by Open Beauty Facts:
    /cgi/search.pl?search_terms=...&json=1
    """
    params = {
        "search_terms": search_terms,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page": page,
        "page_size": page_size,
        "fields": ",".join(OBF_FIELDS),
    }
    r = requests.get(f"{OBF_BASE}/cgi/search.pl", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def normalize_brand(brands: str | None) -> str:
    if not brands:
        return "Unknown"
    # OBF brands can be "Brand1, Brand2"
    b = brands.split(",")[0].strip()
    return b if b else "Unknown"

def extract_products() -> pd.DataFrame:
    rows = []

    for mapped_category, keywords in SEARCH_STREAMS:
        for kw in keywords:
            data = obf_search(kw, page=1, page_size=50)
            products = data.get("products", [])

            for p in products:
                code = p.get("code")
                name = (p.get("product_name") or "").strip()
                if not code or not name:
                    continue

                brand = normalize_brand(p.get("brands"))
                categories_tags = p.get("categories_tags") or []
                categories_text = p.get("categories") or ""

                # Extra guards to keep results relevant (avoid junk/incomplete)
                blob = f"{name} {categories_text} {' '.join(categories_tags)}".lower()
                if mapped_category == "skincare":
                    if not any(x in blob for x in ["skin", "face", "cream", "serum", "clean", "spf", "lotion", "moist"]):
                        continue
                else:  # makeup
                    if not any(x in blob for x in ["lip", "mascara", "foundation", "eyeliner", "blush", "makeup", "palette"]):
                        continue

                ingredients_text = (p.get("ingredients_text") or "").strip()

                rows.append({
                    "product_id": f"obf_{code}",
                    "name": name,
                    "brand": brand,
                    "category": mapped_category,          # skincare | makeup
                    "subcategory": "openbeautyfacts",
                    "price": None,
                    "currency": "USD",
                    "rating": None,
                    "review_count": None,
                    "image_url": p.get("image_front_url") or "",
                    "product_url": p.get("url") or f"{OBF_BASE}/product/{code}",
                    "in_stock": None,
                    "ingredients_text": ingredients_text,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    return df

def write_pipeline_run(engine, run_id, start, end, status, upserted=0, snap=0, error=None):
    sql = """
    INSERT INTO pipeline_runs (run_id, started_at, finished_at, status, rows_upserted, rows_snapshotted, error_message)
    VALUES (:run_id, :start, :end, :status, :upserted, :snap, :error)
    ON CONFLICT (run_id) DO UPDATE SET
      finished_at = EXCLUDED.finished_at,
      status = EXCLUDED.status,
      rows_upserted = EXCLUDED.rows_upserted,
      rows_snapshotted = EXCLUDED.rows_snapshotted,
      error_message = EXCLUDED.error_message;
    """
    with engine.begin() as conn:
        conn.execute(text(sql), {
            "run_id": run_id,
            "start": start,
            "end": end,
            "status": status,
            "upserted": upserted,
            "snap": snap,
            "error": error
        })

def upsert_products(engine, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    sql = """
    INSERT INTO products (
      product_id, name, brand, category, subcategory,
      price, currency, rating, review_count,
      image_url, product_url, ingredients_text, updated_at
    )
    VALUES (
      :product_id, :name, :brand, :category, :subcategory,
      :price, :currency, :rating, :review_count,
      :image_url, :product_url, :ingredients_text, NOW()
    )
    ON CONFLICT (product_id) DO UPDATE SET
      name = EXCLUDED.name,
      brand = EXCLUDED.brand,
      category = EXCLUDED.category,
      subcategory = EXCLUDED.subcategory,
      price = EXCLUDED.price,
      currency = EXCLUDED.currency,
      rating = EXCLUDED.rating,
      review_count = EXCLUDED.review_count,
      image_url = EXCLUDED.image_url,
      product_url = EXCLUDED.product_url,
      ingredients_text = EXCLUDED.ingredients_text,
      updated_at = NOW();
    """
    with engine.begin() as conn:
        conn.execute(text(sql), df.to_dict(orient="records"))
    return len(df)

def snapshot_today(engine, df: pd.DataFrame, today: dt.date) -> int:
    if df.empty:
        return 0

    snap = df[["product_id", "price", "rating", "review_count", "in_stock"]].copy()
    snap["snapshot_date"] = today

    sql = """
    INSERT INTO product_snapshots (snapshot_date, product_id, price, rating, review_count, in_stock)
    VALUES (:snapshot_date, :product_id, :price, :rating, :review_count, :in_stock)
    ON CONFLICT (snapshot_date, product_id) DO UPDATE SET
      price = EXCLUDED.price,
      rating = EXCLUDED.rating,
      review_count = EXCLUDED.review_count,
      in_stock = EXCLUDED.in_stock;
    """
    with engine.begin() as conn:
        conn.execute(text(sql), snap.to_dict(orient="records"))
    return len(snap)

def main():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    run_id = str(uuid.uuid4())
    started_at = dt.datetime.now(dt.UTC)
    write_pipeline_run(engine, run_id, started_at, None, "running")

    try:
        df = extract_products()
        today = dt.date.today()

        upserted = upsert_products(engine, df)
        snap = snapshot_today(engine, df, today)

        finished_at = dt.datetime.now(dt.UTC)
        write_pipeline_run(engine, run_id, started_at, finished_at, "success", upserted, snap)

        print(f"✅ Pipeline SUCCESS | obf_rows={upserted} | snapshot_date={today}")

    except Exception as e:
        finished_at = dt.datetime.now(dt.UTC)
        write_pipeline_run(engine, run_id, started_at, finished_at, "failed", 0, 0, str(e))
        raise

if __name__ == "__main__":
    main()

import re
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="BeautyPulse Dashboard", layout="wide")
st.title("💄 BeautyPulse – Skincare & Makeup Insights")
st.caption("Open Beauty Facts + Data Engineering pipeline + Ingredients analytics + data quality monitoring")

# -----------------------
# DB (Cloud-safe)
# -----------------------
@st.cache_resource
def get_engine():
    db_url = st.secrets["DATABASE_URL"]
    return create_engine(db_url, pool_pre_ping=True)

engine = get_engine()

@st.cache_data(ttl=300)
def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

def download_button_df(label: str, df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")

def link_cta(label: str, url: str):
    if not url:
        return
    if hasattr(st, "link_button"):
        st.link_button(label, url)
    else:
        st.markdown(f"[{label}]({url})")

# -----------------------
# Ingredient helpers
# -----------------------
def tokenize_ingredients(txt: str) -> list[str]:
    if not txt:
        return []
    t = txt.lower()
    t = re.sub(r"\([^)]*\)", " ", t)
    t = re.sub(r"[%\d]", " ", t)
    parts = re.split(r"[;,/]", t)
    out = []
    for p in parts:
        p = p.strip()
        if 2 <= len(p) <= 40:
            out.append(p)
    return out

def flag_alcohol(txt: str) -> bool:
    if not txt:
        return False
    t = txt.lower()
    return any(p in t for p in ["alcohol denat", "ethanol", "isopropyl alcohol", "sd alcohol"])

def flag_fragrance(txt: str) -> bool:
    if not txt:
        return False
    t = txt.lower()
    return any(p in t for p in ["fragrance", "parfum", "aroma"])

def flag_parabens(txt: str) -> bool:
    if not txt:
        return False
    t = txt.lower()
    return any(p in t for p in ["methylparaben", "propylparaben", "butylparaben", "ethylparaben"])

def clean_score(txt: str) -> tuple[int, list[str]]:
    """
    Explainable Clean Score (0–100):
    start 100
    -35 parabens
    -20 drying alcohol
    -10 fragrance
    """
    if not txt:
        return 0, ["No ingredient list available"]

    score = 100
    reasons = []

    if flag_parabens(txt):
        score -= 35
        reasons.append("Parabens detected (-35)")
    if flag_alcohol(txt):
        score -= 20
        reasons.append("Drying alcohol detected (-20)")
    if flag_fragrance(txt):
        score -= 10
        reasons.append("Fragrance detected (-10)")

    score = max(0, min(100, score))
    if not reasons:
        reasons = ["No flagged ingredients found"]

    return score, reasons

def score_badge(score: int) -> str:
    if score >= 85:
        return "✅ Excellent"
    if score >= 70:
        return "🟢 Good"
    if score >= 50:
        return "🟠 Caution"
    return "🔴 High Risk"

# -----------------------
# Navigation
# -----------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "Home (KPIs)",
        "Product Explorer (Cards)",
        "Ingredients Analytics",
        "Ingredient Checker",
        "Data Quality",
        "Compare Products",
        "Brand Insights",
        "Pipeline Status",
    ],
)

# ======================================================
# HOME
# ======================================================
if page == "Home (KPIs)":
    kpis = run_query("""
    SELECT
      COUNT(*) AS total_products,
      COUNT(DISTINCT brand) AS total_brands,
      SUM(CASE WHEN category = 'skincare' THEN 1 ELSE 0 END) AS skincare_products,
      SUM(CASE WHEN category = 'makeup' THEN 1 ELSE 0 END) AS makeup_products,
      SUM(CASE WHEN image_url IS NOT NULL AND image_url <> '' THEN 1 ELSE 0 END) AS with_images,
      SUM(CASE WHEN product_url IS NOT NULL AND product_url <> '' THEN 1 ELSE 0 END) AS with_links,
      SUM(CASE WHEN ingredients_text IS NOT NULL AND ingredients_text <> '' THEN 1 ELSE 0 END) AS with_ingredients,
      MAX(updated_at) AS last_updated
    FROM products;
    """).iloc[0]

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Total Products", int(kpis.total_products))
    c2.metric("Brands", int(kpis.total_brands))
    c3.metric("Skincare", int(kpis.skincare_products))
    c4.metric("Makeup", int(kpis.makeup_products))
    c5.metric("With Images", int(kpis.with_images))
    c6.metric("With Links", int(kpis.with_links))
    c7.metric("With Ingredients", int(kpis.with_ingredients))

    if kpis.last_updated is not None:
        st.write(f"**Last Updated:** {pd.to_datetime(kpis.last_updated).strftime('%Y-%m-%d %H:%M:%S')}")

    st.divider()

    st.subheader("Category Split")
    split_df = run_query("""
    SELECT category, COUNT(*) AS count
    FROM products
    GROUP BY category
    ORDER BY count DESC;
    """)
    st.bar_chart(split_df.set_index("category"))
    download_button_df("⬇️ Download category split CSV", split_df, "category_split.csv")

# ======================================================
# PRODUCT EXPLORER
# ======================================================
elif page == "Product Explorer (Cards)":
    st.subheader("🖼️ Product Explorer (Table + Cards)")

    df = run_query("""
    SELECT name, brand, category, subcategory, image_url, product_url, ingredients_text, updated_at
    FROM products
    ORDER BY updated_at DESC;
    """)

    f1, f2, f3 = st.columns(3)
    categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
    brands = ["All"] + sorted(df["brand"].dropna().unique().tolist())

    selected_category = f1.selectbox("Category", categories)
    selected_brand = f2.selectbox("Brand", brands)
    search_text = f3.text_input("Search product name")

    if selected_category != "All":
        df = df[df["category"] == selected_category]
    if selected_brand != "All":
        df = df[df["brand"] == selected_brand]
    if search_text.strip():
        df = df[df["name"].str.contains(search_text, case=False, na=False)]

    st.caption(f"Showing **{len(df)}** products")
    download_button_df("⬇️ Download filtered products CSV", df.drop(columns=["ingredients_text"], errors="ignore"), "filtered_products.csv")

    view_mode = st.radio("View mode", ["Cards", "Table"], horizontal=True)

    if view_mode == "Table":
        display_df = df.drop(columns=["ingredients_text"], errors="ignore").copy()
        display_df["updated_at"] = pd.to_datetime(display_df["updated_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        display_df.reset_index(drop=True, inplace=True)
        display_df.index = display_df.index + 1
        display_df.index.name = "S.No"
        st.dataframe(display_df[["name", "brand", "category", "subcategory", "updated_at"]], use_container_width=True)
    else:
        max_cards = st.slider("How many cards to show", 6, 60, 18, step=6)
        cards_df = df.copy().reset_index(drop=True).head(max_cards)

        cols = st.columns(3)
        for i, row in cards_df.iterrows():
            with cols[i % 3]:
                img = (row.get("image_url") or "").strip()
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.caption("No image available")

                ing = (row.get("ingredients_text") or "").strip()
                score, reasons = clean_score(ing)

                st.markdown(f"**{row.get('name','')}**")
                st.write(f"Brand: {row.get('brand','')}")
                st.write(f"Category: {row.get('category','')}")
                st.write(f"Clean Score: **{score}** — {score_badge(score)}")
                st.caption("; ".join(reasons))

                url = (row.get("product_url") or "").strip()
                if url:
                    link_cta("Open Product", url)

# ======================================================
# INGREDIENTS ANALYTICS
# ======================================================
elif page == "Ingredients Analytics":
    st.subheader("🧪 Ingredients Analytics (Top ingredients + flags)")

    base = run_query("""
    SELECT name, brand, category, ingredients_text
    FROM products
    WHERE ingredients_text IS NOT NULL AND ingredients_text <> '';
    """)

    if base.empty:
        st.warning("No ingredients_text found. Run pipeline to load ingredients.")
        st.stop()

    base["has_alcohol"] = base["ingredients_text"].apply(flag_alcohol)
    base["has_fragrance"] = base["ingredients_text"].apply(flag_fragrance)
    base["has_parabens"] = base["ingredients_text"].apply(flag_parabens)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Products Analyzed", int(len(base)))
    k2.metric("Alcohol flagged", int(base["has_alcohol"].sum()))
    k3.metric("Fragrance flagged", int(base["has_fragrance"].sum()))
    k4.metric("Parabens flagged", int(base["has_parabens"].sum()))

    st.divider()

    st.subheader("Flags by Category (% of products)")
    flags_by_cat = (
        base.groupby("category")[["has_alcohol", "has_fragrance", "has_parabens"]]
        .mean()
        .reset_index()
    )
    flags_by_cat["has_alcohol"] = (flags_by_cat["has_alcohol"] * 100).round(2)
    flags_by_cat["has_fragrance"] = (flags_by_cat["has_fragrance"] * 100).round(2)
    flags_by_cat["has_parabens"] = (flags_by_cat["has_parabens"] * 100).round(2)
    flags_by_cat.rename(columns={"has_alcohol": "% Alcohol", "has_fragrance": "% Fragrance", "has_parabens": "% Parabens"}, inplace=True)

    flags_by_cat.reset_index(drop=True, inplace=True)
    flags_by_cat.index = flags_by_cat.index + 1
    flags_by_cat.index.name = "S.No"
    st.dataframe(flags_by_cat, use_container_width=True)
    download_button_df("⬇️ Download flags-by-category CSV", flags_by_cat.reset_index(drop=True), "flags_by_category.csv")

    st.divider()

    st.subheader("Top Ingredients (tokenized)")
    cat_choice = st.selectbox("Analyze category", ["All", "skincare", "makeup"])
    temp = base.copy()
    if cat_choice != "All":
        temp = temp[temp["category"] == cat_choice]

    tokens = []
    for txt in temp["ingredients_text"].tolist():
        tokens.extend(tokenize_ingredients(txt))

    top = pd.Series(tokens).value_counts().head(30).reset_index()
    top.columns = ["ingredient", "count"]
    top.index = top.index + 1
    top.index.name = "S.No"
    st.dataframe(top, use_container_width=True)
    download_button_df("⬇️ Download top ingredients CSV", top.reset_index(drop=True), "top_ingredients.csv")

# ======================================================
# INGREDIENT CHECKER
# ======================================================
elif page == "Ingredient Checker":
    st.subheader("🔍 Ingredient Checker (Search + Clean Score)")
    st.caption("Search any ingredient keyword and compare products (lowest risk first).")

    search = st.text_input("Ingredient keyword", placeholder="e.g., niacinamide")
    cat = st.selectbox("Filter category", ["All", "skincare", "makeup"])
    limit = st.slider("Max results", 10, 200, 50, step=10)

    if not search.strip():
        st.info("Enter an ingredient keyword to search.")
        st.stop()

    df = run_query("""
    SELECT name, brand, category, ingredients_text, image_url, product_url
    FROM products
    WHERE ingredients_text IS NOT NULL AND ingredients_text <> ''
      AND LOWER(ingredients_text) LIKE '%' || LOWER(:kw) || '%'
    ORDER BY updated_at DESC
    LIMIT 1000;
    """, params={"kw": search.strip()})

    if df.empty:
        st.warning("No products found for that keyword.")
        st.stop()

    if cat != "All":
        df = df[df["category"] == cat]

    scores = df["ingredients_text"].apply(clean_score)
    df["clean_score"] = [s[0] for s in scores]
    df["reasons"] = ["; ".join(s[1]) for s in scores]
    df["badge"] = df["clean_score"].apply(score_badge)

    df = df.sort_values(["clean_score", "brand", "name"], ascending=[False, True, True]).head(limit).reset_index(drop=True)

    st.write(f"Showing **{len(df)}** matches (higher clean score first).")
    download_button_df("⬇️ Download results CSV", df.drop(columns=["ingredients_text"], errors="ignore"), f"ingredient_search_{search.strip().lower()}.csv")

    cols = st.columns(3)
    for i, row in df.iterrows():
        with cols[i % 3]:
            img = (row.get("image_url") or "").strip()
            if img:
                st.image(img, use_container_width=True)

            st.markdown(f"**{row['name']}**")
            st.write(f"Brand: {row['brand']}")
            st.write(f"Category: {row['category']}")
            st.write(f"Clean Score: **{row['clean_score']}** — {row['badge']}")
            st.caption(row["reasons"])
            url = (row.get("product_url") or "").strip()
            if url:
                link_cta("Open Product", url)

# ======================================================
# DATA QUALITY
# ======================================================
elif page == "Data Quality":
    st.subheader("✅ Data Quality Dashboard")
    st.caption("Coverage metrics to validate completeness of brand/image/link/ingredients fields.")

    dq = run_query("""
    SELECT
      ROUND(100.0 * SUM(CASE WHEN brand IS NULL OR brand = '' OR brand = 'Unknown' THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2) AS pct_missing_brand,
      ROUND(100.0 * SUM(CASE WHEN image_url IS NULL OR image_url = '' THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2) AS pct_missing_image,
      ROUND(100.0 * SUM(CASE WHEN product_url IS NULL OR product_url = '' THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2) AS pct_missing_link,
      ROUND(100.0 * SUM(CASE WHEN ingredients_text IS NULL OR ingredients_text = '' THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2) AS pct_missing_ingredients
    FROM products;
    """).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("% Missing Brand", f"{dq.pct_missing_brand}%")
    c2.metric("% Missing Image", f"{dq.pct_missing_image}%")
    c3.metric("% Missing Link", f"{dq.pct_missing_link}%")
    c4.metric("% Missing Ingredients", f"{dq.pct_missing_ingredients}%")

    st.divider()

    st.subheader("Coverage by Category")
    cat_dq = run_query("""
    SELECT
      category,
      ROUND(100.0 * SUM(CASE WHEN brand IS NULL OR brand = '' OR brand = 'Unknown' THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2) AS pct_missing_brand,
      ROUND(100.0 * SUM(CASE WHEN image_url IS NULL OR image_url = '' THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2) AS pct_missing_image,
      ROUND(100.0 * SUM(CASE WHEN product_url IS NULL OR product_url = '' THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2) AS pct_missing_link,
      ROUND(100.0 * SUM(CASE WHEN ingredients_text IS NULL OR ingredients_text = '' THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 2) AS pct_missing_ingredients
    FROM products
    GROUP BY category
    ORDER BY category;
    """)

    cat_dq.reset_index(drop=True, inplace=True)
    cat_dq.index = cat_dq.index + 1
    cat_dq.index.name = "S.No"
    st.dataframe(cat_dq, use_container_width=True)
    download_button_df("⬇️ Download data quality CSV", cat_dq.reset_index(drop=True), "data_quality.csv")

# ======================================================
# COMPARE PRODUCTS
# ======================================================
elif page == "Compare Products":
    st.subheader("🆚 Compare Products")
    st.caption("Pick two products and compare clean score, flags, and ingredient overlap.")

    options = run_query("""
    SELECT product_id, name, brand, category, ingredients_text
    FROM products
    WHERE name IS NOT NULL AND name <> ''
    ORDER BY updated_at DESC
    LIMIT 500;
    """)

    options["label"] = options["name"] + " — " + options["brand"] + " (" + options["category"] + ")"
    labels = options["label"].tolist()

    left, right = st.columns(2)
    choice_a = left.selectbox("Product A", labels, index=0)
    choice_b = right.selectbox("Product B", labels, index=min(1, len(labels) - 1))

    row_a = options[options["label"] == choice_a].iloc[0]
    row_b = options[options["label"] == choice_b].iloc[0]

    ing_a = (row_a["ingredients_text"] or "").strip()
    ing_b = (row_b["ingredients_text"] or "").strip()

    score_a, reasons_a = clean_score(ing_a)
    score_b, reasons_b = clean_score(ing_b)

    tok_a = set(tokenize_ingredients(ing_a))
    tok_b = set(tokenize_ingredients(ing_b))
    overlap = tok_a.intersection(tok_b)
    overlap_pct = 0.0 if (len(tok_a) + len(tok_b)) == 0 else (2 * len(overlap) / (len(tok_a) + len(tok_b))) * 100

    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Clean Score A", score_a)
    c2.metric("Clean Score B", score_b)
    c3.metric("Ingredient Overlap %", f"{overlap_pct:.1f}%")

    st.subheader("Reasons")
    col1, col2 = st.columns(2)
    col1.write("**A:** " + "; ".join(reasons_a))
    col2.write("**B:** " + "; ".join(reasons_b))

    st.subheader("Shared Ingredients (Top 50)")
    shared = sorted(list(overlap))[:50]
    shared_df = pd.DataFrame({"shared_ingredients": shared})
    shared_df.index = shared_df.index + 1
    shared_df.index.name = "S.No"
    st.dataframe(shared_df, use_container_width=True)

# ======================================================
# BRAND INSIGHTS
# ======================================================
elif page == "Brand Insights":
    st.subheader("🏷️ Brand Insights")

    brand_df = run_query("""
    SELECT
      brand,
      category,
      COUNT(*) AS product_count,
      SUM(CASE WHEN image_url IS NOT NULL AND image_url <> '' THEN 1 ELSE 0 END) AS with_images
    FROM products
    GROUP BY brand, category
    HAVING COUNT(*) >= 3
    ORDER BY product_count DESC
    LIMIT 50;
    """)

    brand_df.reset_index(drop=True, inplace=True)
    brand_df.index = brand_df.index + 1
    brand_df.index.name = "S.No"
    st.dataframe(brand_df, use_container_width=True)

    chart_df = (
        brand_df.groupby("brand", as_index=False)["product_count"]
        .sum()
        .sort_values("product_count", ascending=False)
        .head(20)
    )
    st.bar_chart(chart_df.set_index("brand"))

# ======================================================
# PIPELINE STATUS
# ======================================================
elif page == "Pipeline Status":
    st.subheader("🛠 Pipeline Status")

    runs_df = run_query("""
    SELECT
      run_id,
      started_at,
      finished_at,
      status,
      rows_upserted,
      rows_snapshotted,
      error_message
    FROM pipeline_runs
    ORDER BY started_at DESC
    LIMIT 20;
    """)

    if runs_df.empty:
        st.info("No pipeline runs found yet.")
        st.stop()

    for col in ["started_at", "finished_at"]:
        runs_df[col] = pd.to_datetime(runs_df[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    runs_df.reset_index(drop=True, inplace=True)
    runs_df.index = runs_df.index + 1
    runs_df.index.name = "Run #"
    st.dataframe(runs_df, use_container_width=True)

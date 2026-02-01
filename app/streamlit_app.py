import re
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="BeautyPulse | Beauty Analytics Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .section-title {font-size: 1.1rem; font-weight: 700; margin-top: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💄 BeautyPulse – Beauty Analytics Dashboard")
st.caption("Real cosmetics data + cloud Postgres + Streamlit. Built for recruiter-ready storytelling.")

# =========================
# DB (Streamlit Cloud-safe)
# =========================
@st.cache_resource
def get_engine():
    return create_engine(st.secrets["DATABASE_URL"], pool_pre_ping=True)

engine = get_engine()

@st.cache_data(ttl=300)
def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

def download_df(label: str, df: pd.DataFrame, filename: str):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv")

def link_cta(label: str, url: str):
    if not url:
        return
    if hasattr(st, "link_button"):
        st.link_button(label, url)
    else:
        st.markdown(f"[{label}]({url})")

# =========================
# INGREDIENT / FLAGS
# =========================
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

def badge(score: int) -> str:
    if score >= 85:
        return "✅ Excellent"
    if score >= 70:
        return "🟢 Good"
    if score >= 50:
        return "🟠 Caution"
    return "🔴 High Risk"

# =========================
# LOAD BASE DATA (cached)
# =========================
@st.cache_data(ttl=300)
def load_products() -> pd.DataFrame:
    return run_query(
        """
        SELECT
          product_id,
          name,
          brand,
          category,
          subcategory,
          image_url,
          product_url,
          ingredients_text,
          updated_at
        FROM products
        ORDER BY updated_at DESC;
        """
    )

df_all = load_products()
if df_all.empty:
    st.error("No data in `products` table. Run the pipeline first.")
    st.stop()

# =========================
# GLOBAL FILTERS
# =========================
st.sidebar.header("Filters")

cat_options = ["All"] + sorted(df_all["category"].dropna().unique().tolist())
brand_options = ["All"] + sorted(df_all["brand"].dropna().unique().tolist())

sel_cat = st.sidebar.selectbox("Category", cat_options, index=0)
sel_brand = st.sidebar.selectbox("Brand", brand_options, index=0)
search_name = st.sidebar.text_input("Search product name")

df = df_all.copy()
if sel_cat != "All":
    df = df[df["category"] == sel_cat]
if sel_brand != "All":
    df = df[df["brand"] == sel_brand]
if search_name.strip():
    df = df[df["name"].str.contains(search_name, case=False, na=False)]

st.sidebar.caption(f"Filtered rows: {len(df)}")

# =========================
# NAVIGATION
# =========================
page = st.sidebar.radio(
    "Pages",
    ["Overview", "Insights (Executive Summary)", "Ingredients", "Explore Products", "Compare", "Pipeline"],
)

# =========================
# OVERVIEW
# =========================
if page == "Overview":
    st.subheader("📌 Overview")

    total_products = len(df)
    total_brands = df["brand"].nunique()
    skincare_count = int((df["category"] == "skincare").sum())
    makeup_count = int((df["category"] == "makeup").sum())

    with_images = int(df["image_url"].fillna("").ne("").sum())
    with_links = int(df["product_url"].fillna("").ne("").sum())
    with_ingredients = int(df["ingredients_text"].fillna("").ne("").sum())
    last_updated = pd.to_datetime(df["updated_at"], errors="coerce").max()

    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Products", total_products)
    k2.metric("Brands", total_brands)
    k3.metric("Skincare", skincare_count)
    k4.metric("Makeup", makeup_count)
    k5.metric("Images", with_images)
    k6.metric("Links", with_links)
    k7.metric("Ingredients", with_ingredients)

    if pd.notna(last_updated):
        st.caption(f"Latest update in selection: **{last_updated.strftime('%Y-%m-%d %H:%M:%S')}**")

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="section-title">Category Distribution</div>', unsafe_allow_html=True)
        cat_df = df.groupby("category", as_index=False).size().rename(columns={"size": "count"})
        st.bar_chart(cat_df.set_index("category"))
        download_df("⬇️ Download category distribution", cat_df, "category_distribution.csv")

    with right:
        st.markdown('<div class="section-title">Data Quality Coverage (%)</div>', unsafe_allow_html=True)
        dq = pd.DataFrame(
            [
                {
                    "metric": "Missing brand",
                    "pct": round(100 * (df["brand"].fillna("").isin(["", "Unknown"]).mean()), 2),
                },
                {"metric": "Missing image", "pct": round(100 * (df["image_url"].fillna("").eq("").mean()), 2)},
                {"metric": "Missing link", "pct": round(100 * (df["product_url"].fillna("").eq("").mean()), 2)},
                {
                    "metric": "Missing ingredients",
                    "pct": round(100 * (df["ingredients_text"].fillna("").eq("").mean()), 2),
                },
            ]
        )
        st.dataframe(dq, use_container_width=True)
        download_df("⬇️ Download data quality", dq, "data_quality.csv")

    st.divider()

    st.markdown('<div class="section-title">Top Brands (by product count)</div>', unsafe_allow_html=True)
    top_brands = df.groupby("brand", as_index=False).size().rename(columns={"size": "count"})
    top_brands = top_brands.sort_values("count", ascending=False).head(20)
    st.bar_chart(top_brands.set_index("brand"))
    download_df("⬇️ Download top brands", top_brands, "top_brands.csv")

# =========================
# EXECUTIVE INSIGHTS
# =========================
elif page == "Insights (Executive Summary)":
    st.subheader("📊 Executive Insights")
    st.caption("Auto-generated insights derived from the current filtered selection.")

    base = df[df["ingredients_text"].fillna("").ne("")].copy()
    if base.empty:
        st.warning("No ingredient data available for insights in this selection.")
        st.stop()

    # Clean scores + flags
    sc = base["ingredients_text"].apply(clean_score)
    base["clean_score"] = [x[0] for x in sc]
    base["has_alcohol"] = base["ingredients_text"].apply(flag_alcohol)
    base["has_fragrance"] = base["ingredients_text"].apply(flag_fragrance)
    base["has_parabens"] = base["ingredients_text"].apply(flag_parabens)

    pct_fragrance = round(100 * base["has_fragrance"].mean(), 1)
    pct_alcohol = round(100 * base["has_alcohol"].mean(), 1)
    pct_parabens = round(100 * base["has_parabens"].mean(), 1)

    st.markdown("### 🔍 Key Findings")
    st.markdown(
        f"""
- 📌 **{pct_fragrance}%** of products contain **fragrance/parfum**, which may be unsuitable for **sensitive-skin** users.
- ⚠️ **{pct_alcohol}%** of products contain **drying alcohol**, a common irritant in some formulations.
- 🚨 **{pct_parabens}%** of products contain **parabens**, frequently avoided in clean-beauty preferences.
"""
    )

    skincare = base[base["category"] == "skincare"]
    makeup = base[base["category"] == "makeup"]
    if not skincare.empty and not makeup.empty:
        alc_sk = round(100 * skincare["has_alcohol"].mean(), 1)
        alc_mu = round(100 * makeup["has_alcohol"].mean(), 1)
        ratio = round(alc_mu / max(alc_sk, 1), 1)
        st.markdown(f"- 🧴 **Makeup uses alcohol ~{ratio}× more frequently** than skincare (within this selection).")

    st.divider()

    st.markdown("### 🧼 Clean Score Distribution")
    bins = pd.cut(
        base["clean_score"],
        bins=[-1, 49, 69, 84, 100],
        labels=["High Risk (<50)", "Caution (50–69)", "Good (70–84)", "Excellent (85+)"],
    )
    dist = bins.value_counts().sort_index().reset_index()
    dist.columns = ["Band", "Products"]
    st.bar_chart(dist.set_index("Band"))
    download_df("⬇️ Download clean score distribution", dist, "clean_score_distribution.csv")

    st.divider()

    st.markdown("### 🏆 Cleanest Brands (min 5 products)")
    brand_scores = (
        base.groupby("brand", as_index=False)
        .agg(avg_clean_score=("clean_score", "mean"), products=("clean_score", "count"))
    )
    brand_scores = brand_scores[brand_scores["products"] >= 5]
    brand_scores = brand_scores.sort_values("avg_clean_score", ascending=False).head(10)
    brand_scores["avg_clean_score"] = brand_scores["avg_clean_score"].round(1)
    brand_scores = brand_scores.reset_index(drop=True)
    brand_scores.index = brand_scores.index + 1
    brand_scores.index.name = "Rank"
    st.dataframe(brand_scores, use_container_width=True)
    download_df("⬇️ Download cleanest brands", brand_scores.reset_index(drop=True), "cleanest_brands.csv")

# =========================
# INGREDIENTS
# =========================
elif page == "Ingredients":
    st.subheader("🧪 Ingredients Insights")

    base = df[df["ingredients_text"].fillna("").ne("")].copy()
    if base.empty:
        st.warning("No products with ingredients in this selection.")
        st.stop()

    base["has_alcohol"] = base["ingredients_text"].apply(flag_alcohol)
    base["has_fragrance"] = base["ingredients_text"].apply(flag_fragrance)
    base["has_parabens"] = base["ingredients_text"].apply(flag_parabens)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products w/ Ingredients", len(base))
    c2.metric("Alcohol flagged", int(base["has_alcohol"].sum()))
    c3.metric("Fragrance flagged", int(base["has_fragrance"].sum()))
    c4.metric("Parabens flagged", int(base["has_parabens"].sum()))

    st.divider()

    st.markdown('<div class="section-title">Flag Rates (%)</div>', unsafe_allow_html=True)
    rates = pd.DataFrame(
        [
            {"flag": "Alcohol", "pct": round(100 * base["has_alcohol"].mean(), 2)},
            {"flag": "Fragrance", "pct": round(100 * base["has_fragrance"].mean(), 2)},
            {"flag": "Parabens", "pct": round(100 * base["has_parabens"].mean(), 2)},
        ]
    )
    st.bar_chart(rates.set_index("flag"))
    download_df("⬇️ Download flag rates", rates, "flag_rates.csv")

    st.divider()

    st.markdown('<div class="section-title">Top Ingredients</div>', unsafe_allow_html=True)
    tokens = []
    for txt in base["ingredients_text"].tolist():
        tokens.extend(tokenize_ingredients(txt))

    top = pd.Series(tokens).value_counts().head(30).reset_index()
    top.columns = ["ingredient", "count"]
    st.dataframe(top, use_container_width=True)
    download_df("⬇️ Download top ingredients", top, "top_ingredients.csv")

    st.divider()

    st.markdown('<div class="section-title">Ingredient Checker</div>', unsafe_allow_html=True)
    kw = st.text_input("Search ingredient keyword (e.g., niacinamide, retinol, fragrance)")
    if kw.strip():
        hit = base[base["ingredients_text"].str.contains(kw, case=False, na=False)].copy()
        if hit.empty:
            st.info("No matches found.")
        else:
            hit["clean_score"] = hit["ingredients_text"].apply(lambda x: clean_score(x)[0])
            hit = hit.sort_values("clean_score", ascending=False).head(50)

            st.caption(f"Matches: {len(hit)} (showing top 50 by clean score)")
            download_df("⬇️ Download ingredient search results", hit.drop(columns=["ingredients_text"]), f"ingredient_search_{kw.lower()}.csv")

            cols = st.columns(3)
            for i, row in hit.reset_index(drop=True).iterrows():
                with cols[i % 3]:
                    img = (row.get("image_url") or "").strip()
                    if img:
                        st.image(img, use_container_width=True)
                    st.markdown(f"**{row['name']}**")
                    st.write(f"Brand: {row['brand']}")
                    st.write(f"Category: {row['category']}")
                    sc2, reasons = clean_score(row["ingredients_text"])
                    st.write(f"Clean Score: **{sc2}** — {badge(sc2)}")
                    st.caption("; ".join(reasons))
                    link_cta("Open Product", (row.get("product_url") or "").strip())

# =========================
# EXPLORE
# =========================
elif page == "Explore Products":
    st.subheader("🖼️ Explore Products")
    st.caption("Use filters on the left to narrow down. Switch between Cards and Table. Export results anytime.")

    download_df("⬇️ Download filtered products", df.drop(columns=["ingredients_text"], errors="ignore"), "filtered_products.csv")

    mode = st.radio("View mode", ["Cards", "Table"], horizontal=True)

    if mode == "Table":
        show = df.copy()
        show["updated_at"] = pd.to_datetime(show["updated_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        show = show[["name", "brand", "category", "subcategory", "updated_at"]].reset_index(drop=True)
        show.index = show.index + 1
        show.index.name = "S.No"
        st.dataframe(show, use_container_width=True)
    else:
        max_cards = st.slider("Cards to show", 6, 60, 18, step=6)
        cards = df.reset_index(drop=True).head(max_cards)

        cols = st.columns(3)
        for i, row in cards.iterrows():
            with cols[i % 3]:
                img = (row.get("image_url") or "").strip()
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.caption("No image available")

                ing = (row.get("ingredients_text") or "")
                sc3, reasons = clean_score(ing)

                st.markdown(f"**{row.get('name','')}**")
                st.write(f"Brand: {row.get('brand','')}")
                st.write(f"Category: {row.get('category','')}")
                st.write(f"Clean Score: **{sc3}** — {badge(sc3)}")
                st.caption("; ".join(reasons))
                link_cta("Open Product", (row.get("product_url") or "").strip())

# =========================
# COMPARE
# =========================
elif page == "Compare":
    st.subheader("🆚 Compare Two Products")
    st.caption("Choose two products from the current filtered selection.")

    options = df[df["name"].fillna("").ne("")].copy().head(500)
    options["label"] = (
        options["name"]
        + " — "
        + options["brand"].fillna("Unknown")
        + " ("
        + options["category"].fillna("Unknown")
        + ")"
    )
    labels = options["label"].tolist()

    if len(labels) < 2:
        st.warning("Need at least 2 products in the filtered selection.")
        st.stop()

    c1, c2 = st.columns(2)
    a = c1.selectbox("Product A", labels, index=0)
    b = c2.selectbox("Product B", labels, index=1)

    ra = options[options["label"] == a].iloc[0]
    rb = options[options["label"] == b].iloc[0]

    ia = (ra.get("ingredients_text") or "")
    ib = (rb.get("ingredients_text") or "")

    sa, rea = clean_score(ia)
    sb, reb = clean_score(ib)

    ta = set(tokenize_ingredients(ia))
    tb = set(tokenize_ingredients(ib))
    overlap = ta.intersection(tb)
    overlap_pct = 0.0 if (len(ta) + len(tb)) == 0 else (2 * len(overlap) / (len(ta) + len(tb))) * 100

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Clean Score A", sa)
    m2.metric("Clean Score B", sb)
    m3.metric("Ingredient Overlap %", f"{overlap_pct:.1f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("### A Reasons")
        st.write("; ".join(rea))
    with right:
        st.markdown("### B Reasons")
        st.write("; ".join(reb))

    st.markdown("### Shared Ingredients (top 50)")
    shared = sorted(list(overlap))[:50]
    st.dataframe(pd.DataFrame({"shared_ingredients": shared}), use_container_width=True)

# =========================
# PIPELINE
# =========================
elif page == "Pipeline":
    st.subheader("🛠 Pipeline Status")
    runs = run_query(
        """
        SELECT run_id, started_at, finished_at, status, rows_upserted, rows_snapshotted, error_message
        FROM pipeline_runs
        ORDER BY started_at DESC
        LIMIT 30;
        """
    )
    if runs.empty:
        st.info("No pipeline runs found yet.")
        st.stop()

    for col in ["started_at", "finished_at"]:
        runs[col] = pd.to_datetime(runs[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    runs = runs.reset_index(drop=True)
    runs.index = runs.index + 1
    runs.index.name = "Run #"
    st.dataframe(runs, use_container_width=True)
    download_df("⬇️ Download pipeline runs", runs.reset_index(drop=True), "pipeline_runs.csv")

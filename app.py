from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Flame in Freefall", layout="wide")
BASE = Path(__file__).parent


@st.cache_data
def load_tables():
    """Load every '...Experimental table...' CSV into a dict of DataFrames."""
    tables = {}
    seen = set()
    for path in sorted(BASE.rglob("*Experimental table*.csv")):
        if "(1)" in path.name or path.name in seen:  # skip duplicate downloads
            continue
        seen.add(path.name)
        stem = path.stem
        psi_id = stem.split("_")[0]
        label = stem.split("Experimental table_", 1)[-1]
        try:
            df = pd.read_csv(path, encoding_errors="replace", on_bad_lines="skip")
        except Exception:
            continue
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        df.insert(0, "investigation", label)
        df.insert(1, "psi_id", psi_id)
        tables[f"{label} ({psi_id})"] = df
    return tables


def safe(df):
    """Make mixed-type columns display without errors."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str).replace("nan", "")
    return out


tables = load_tables()

st.title("Flame in Freefall")
st.caption("SI-powered fire safety insights from microgravity combustion data")
st.caption("Data: NASA Physical Science Informatics (PSI). Verify important values against the source.")

if not tables:
    st.error("No 'Experimental table' CSV files found next to app.py.")
    st.stop()

names = list(tables)
chosen = st.sidebar.multiselect("Investigations", names, default=names)
if not chosen:
    st.warning("Pick at least one investigation in the sidebar.")
    st.stop()

combined = pd.concat([tables[n] for n in chosen], ignore_index=True, sort=False)

tab_over, tab_explore, tab_compare, tab_chart = st.tabs(
    ["Overview", "Explore", "Compare", "Chart"]
)

with tab_over:
    summary = pd.DataFrame(
        {
            "Investigation": chosen,
            "Rows": [len(tables[n]) for n in chosen],
            "Columns": [tables[n].shape[1] for n in chosen],
        }
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Investigations", len(chosen))
    c2.metric("Total rows", len(combined))
    c3.metric("Distinct columns", combined.shape[1])
    st.dataframe(summary, use_container_width=True, hide_index=True)

with tab_explore:
    pick = st.selectbox("Investigation", chosen, key="explore_pick")
    df = tables[pick]
    query = st.text_input("Search all columns")
    if query:
        mask = (
            df.astype(str)
            .apply(lambda col: col.str.contains(query, case=False, na=False))
            .any(axis=1)
        )
        df = df[mask]
    st.write(f"{len(df)} rows")
    st.dataframe(safe(df), use_container_width=True)

with tab_compare:
    left, right = st.columns(2)
    a = left.selectbox("Investigation A", chosen, key="cmp_a")
    b = right.selectbox("Investigation B", chosen, index=min(1, len(chosen) - 1), key="cmp_b")
    cols_a = set(tables[a].columns) - {"investigation", "psi_id"}
    cols_b = set(tables[b].columns) - {"investigation", "psi_id"}
    st.write("Columns in both:", ", ".join(sorted(cols_a & cols_b)) or "none")
    st.write("Only in A:", ", ".join(sorted(cols_a - cols_b)) or "none")
    st.write("Only in B:", ", ".join(sorted(cols_b - cols_a)) or "none")
    left.dataframe(safe(tables[a]), use_container_width=True)
    right.dataframe(safe(tables[b]), use_container_width=True)

with tab_chart:
    numeric_cols = []
    for col in combined.columns:
        if col in ("investigation", "psi_id"):
            continue
        values = pd.to_numeric(combined[col], errors="coerce")
        if values.notna().sum() >= 3:
            numeric_cols.append(col)
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns to chart in the selected investigations.")
    else:
        x = st.selectbox("X axis", numeric_cols, key="x_axis")
        y = st.selectbox("Y axis", numeric_cols, index=1, key="y_axis")
        plot_df = combined[["investigation", x, y]].copy()
        plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
        plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
        plot_df = plot_df.dropna()
        if plot_df.empty:
            st.info("No rows have both of those values. Try a different pair.")
        else:
            fig = px.scatter(plot_df, x=x, y=y, color="investigation")
            st.plotly_chart(fig, use_container_width=True)

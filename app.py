from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_mic_recorder import speech_to_text

st.set_page_config(page_title="Flame in Freefall", layout="wide")

try:
    LAMBDA_URL = st.secrets["LAMBDA_URL"]
    API_HEADERS = {"x-app-token": st.secrets["APP_TOKEN"]}
except Exception:
    st.error("Missing secrets. Add LAMBDA_URL and APP_TOKEN to .streamlit/secrets.toml.")
    st.stop()


def call_api(params):
    r = requests.get(LAMBDA_URL, params=params, headers=API_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600)
def load_tables():
    """Ask the Lambda function for every table stored in S3."""
    files = call_api({"action": "list"})["files"]
    tables = {}
    for key in files:
        rows = call_api({"action": "table", "key": key})["rows"]
        df = pd.DataFrame(rows)
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        stem = Path(key).stem
        psi_id = stem.split("_")[0]
        label = stem.split("Experimental table_", 1)[-1]
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


try:
    tables = load_tables()
except Exception as e:
    st.error(f"Could not load data from AWS: {e}")
    st.stop()

st.title("Flame in Freefall")
st.caption("SI-powered fire safety insights from microgravity combustion data")
st.caption("Data: NASA Physical Science Informatics (PSI). Verify important values against the source.")

if not tables:
    st.error("The Lambda function returned no tables. Check the S3 data folder.")
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
    # Voice search: the browser turns speech into text, which fills the search box.
    # Works in Chrome, Edge and Safari. Not in Firefox.
    spoken = speech_to_text(
        language="en",
        start_prompt="🎤 Speak to search",
        stop_prompt="⏹ Stop",
        just_once=True,
        key="voice_search",
    )
    if spoken:
        st.session_state["search_text"] = spoken
    query = st.text_input("Search all columns", key="search_text")
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

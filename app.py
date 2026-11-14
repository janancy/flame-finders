import pandas as pd
import streamlit as st

st.set_page_config(page_title="Flame in Freefall", layout="wide")
st.title("Flame in Freefall")
st.caption("SI-powered fire safety insights from microgravity combustion data")

df = pd.read_csv("data/investigations.csv")

if df.empty:
    st.info("No investigations yet. Add rows to data/investigations.csv.")
else:
    categories = sorted(df["category"].dropna().unique())
    chosen = st.multiselect("Safety category", categories, default=categories)
    filtered = df[df["category"].isin(chosen)]
    st.dataframe(filtered, use_container_width=True)

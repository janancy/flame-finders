import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_mic_recorder import speech_to_text

st.set_page_config(page_title="FLAME", layout="wide")

try:
    LAMBDA_URL = st.secrets["LAMBDA_URL"]
    API_HEADERS = {"x-app-token": st.secrets["APP_TOKEN"]}
except Exception as e:
    st.error(
        f"Missing secrets ({type(e).__name__}). "
        "Add LAMBDA_URL and APP_TOKEN to .streamlit/secrets.toml."
    )
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


@st.cache_data(ttl=600)
def key_map():
    """Map each investigation name to its file path in S3."""
    out = {}
    for key in call_api({"action": "list"})["files"]:
        stem = Path(key).stem
        label = stem.split("Experimental table_", 1)[-1]
        out[f"{label} ({stem.split('_')[0]})"] = key
    return out


def safe(df):
    """Make mixed-type columns display without errors."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str).replace("nan", "")
    return out


def speak_button(text, auto=False):
    """Read aloud / Stop buttons using the browser's built-in voice."""
    clean = re.sub(r"[*_`#>]", "", text)
    js_text = json.dumps(clean).replace("</", "<\\/")
    auto_js = "speak();" if auto else ""
    components.html(
        f"""
        <div style="font-family:sans-serif">
          <button id="play">🔊 Read aloud</button>
          <button id="stop">⏹ Stop</button>
        </div>
        <script>
          const text = {js_text};
          function speak() {{
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(text);
            u.lang = "en-US";
            window.speechSynthesis.speak(u);
          }}
          document.getElementById("play").onclick = speak;
          document.getElementById("stop").onclick = function () {{
            window.speechSynthesis.cancel();
          }};
          {auto_js}
        </script>
        """,
        height=45,
    )


try:
    tables = load_tables()
except Exception as e:
    st.error(f"Could not load data from AWS: {e}")
    st.stop()

st.title("FLAME")
st.caption("Flame in Freefall: SI-powered fire safety insights from microgravity combustion data")
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

tab_over, tab_explore, tab_compare, tab_chart, tab_predict = st.tabs(
    ["Overview", "Explore", "Compare", "Chart", "Predict"]
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

with tab_predict:
    st.subheader("What-if trend explorer")
    st.warning(
        "Exploratory only. This fits a simple trend line to the selected NASA experiments. "
        "It is not a validated fire model and must not be used for safety decisions."
    )
    st.caption("For more meaningful trends, select one investigation (or a few similar ones) in the sidebar.")

    p_cols = []
    for col in combined.columns:
        if col in ("investigation", "psi_id"):
            continue
        if pd.to_numeric(combined[col], errors="coerce").notna().sum() >= 8:
            p_cols.append(col)

    if len(p_cols) < 2:
        st.info("Need at least two numeric columns with 8 or more values in the selected investigations.")
    else:
        pc1, pc2, pc3 = st.columns(3)
        x_col = pc1.selectbox("Input (X): what you change", p_cols, key="pred_x")
        y_col = pc2.selectbox("Output (Y): what you predict", p_cols, index=1, key="pred_y")
        shape = pc3.radio("Trend type", ["Straight line", "Curve"], horizontal=True, key="pred_shape")
        deg = 1 if shape == "Straight line" else 2

        if x_col == y_col:
            st.info("Pick two different columns.")
        else:
            data = combined[["investigation", x_col, y_col]].copy()
            data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
            data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
            data = data.dropna()
            n = len(data)
            xs = data[x_col].to_numpy(dtype=float)
            ys = data[y_col].to_numpy(dtype=float)

            if n < 8 or len(np.unique(xs)) <= deg + 1:
                st.info(
                    f"Only {n} usable points with {len(np.unique(xs))} distinct X values. "
                    "Need at least 8 points and more distinct X values to fit a trend."
                )
            else:
                coef = np.polyfit(xs, ys, deg)
                fitted = np.polyval(coef, xs)
                ss_res = float(np.sum((ys - fitted) ** 2))
                ss_tot = float(np.sum((ys - ys.mean()) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

                errs = []
                for i in range(n):
                    keep = np.arange(n) != i
                    if len(np.unique(xs[keep])) > deg:
                        c = np.polyfit(xs[keep], ys[keep], deg)
                        errs.append(ys[i] - np.polyval(c, xs[i]))
                loo = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")

                x_min, x_max = float(xs.min()), float(xs.max())
                new_x = st.number_input(f"What-if value of {x_col}", value=x_max, key="pred_new_x")
                pred = float(np.polyval(coef, new_x))
                outside = new_x < x_min or new_x > x_max

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Points used", n)
                m2.metric("R² (fit)", f"{r2:.2f}")
                m3.metric("Typical error", f"{loo:.3g}")
                m4.metric("Prediction", f"{pred:.3g}")
                st.caption(
                    f"Prediction is for **{y_col}** when **{x_col}** = {new_x}. "
                    "Typical error is leave-one-out RMSE."
                )

                if outside:
                    st.error(
                        f"{new_x} is outside the tested range ({x_min:.3g} to {x_max:.3g}). "
                        "This is extrapolation and is likely unreliable."
                    )
                elif r2 < 0.5:
                    st.warning("Weak fit: the trend explains little of the variation. Treat this prediction with caution.")

                grid = np.linspace(min(x_min, new_x), max(x_max, new_x), 120)
                line = np.polyval(coef, grid)
                fig = go.Figure()
                for name, grp in data.groupby("investigation"):
                    fig.add_trace(go.Scatter(x=grp[x_col], y=grp[y_col], mode="markers", name=name))
                fig.add_trace(go.Scatter(x=grid, y=line, mode="lines", name="Trend"))
                if not np.isnan(loo):
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([grid, grid[::-1]]),
                            y=np.concatenate([line + 1.96 * loo, (line - 1.96 * loo)[::-1]]),
                            fill="toself",
                            mode="lines",
                            line=dict(width=0),
                            opacity=0.2,
                            name="Approx. range",
                            hoverinfo="skip",
                        )
                    )
                fig.add_trace(
                    go.Scatter(
                        x=[new_x],
                        y=[pred],
                        mode="markers",
                        name="What-if prediction",
                        marker=dict(size=14, symbol="star"),
                    )
                )
                fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Approx. range is ±1.96 × leave-one-out error: a rough guide, not a guarantee.")

st.divider()
st.subheader("Ask the data")
st.caption("Type or speak a question. Tip: select only a few investigations in the sidebar for better answers.")

auto_speak = st.toggle("Speak answers automatically (your browser may block this; use Read aloud if so)")
spoken_q = speech_to_text(
    language="en",
    start_prompt="🎤 Ask by voice",
    stop_prompt="⏹ Stop",
    just_once=True,
    key="voice_chat",
)

if "chat" not in st.session_state:
    st.session_state.chat = []

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m["role"] == "assistant":
            speak_button(m["content"])

typed_q = st.chat_input("Ask about the selected investigations")
prompt = typed_q or spoken_q
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    keys = [key_map()[n] for n in chosen if n in key_map()]
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(
                    LAMBDA_URL,
                    json={"question": prompt, "keys": keys},
                    headers=API_HEADERS,
                    timeout=40,
                )
                r.raise_for_status()
                answer = r.json()["answer"]
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"
        st.write(answer)
        speak_button(answer, auto=auto_speak)
    st.session_state.chat.append({"role": "assistant", "content": answer})

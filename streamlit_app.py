"""Inference Optimization Lab: results dashboard.

Run locally:   streamlit run app/app.py
It reads the CSV files that the notebook writes to results/ (nothing is hard-coded or estimated here).
"""
import glob
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Inference Optimization Lab", layout="wide")

HOURS_PER_MONTH = 730


def candidate_dirs():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get("RESULTS_DIR"), os.path.join(here, "results"),
             os.path.join(here, "..", "results"), "results"]
    return [os.path.abspath(p) for p in cands if p]


def find_results_dir():
    for p in candidate_dirs():
        if glob.glob(os.path.join(p, "load_*.csv")):
            return p
    return None


@st.cache_data
def load_all(results_dir):
    loads, evals = {}, {}
    for p in sorted(glob.glob(os.path.join(results_dir, "load_*.csv"))):
        name = os.path.basename(p)[5:-4]
        if name != "test":
            loads[name] = pd.read_csv(p)
    for p in sorted(glob.glob(os.path.join(results_dir, "eval_*.csv"))):
        name = os.path.basename(p)[5:-4]
        if name != "test":
            evals[name] = pd.read_csv(p)
    return loads, evals


def build_summary(loads, evals, gpu_price):
    rows = []
    for name, l in loads.items():
        c16 = l[l.concurrency == 16]
        tok = float(c16.tokens_per_sec.mean()) if len(c16) else float("nan")
        r = {
            "config": name,
            "tok_s": tok,
            "ttft_p50_ms": float(c16.ttft_ms_p50.mean()) if len(c16) else float("nan"),
            "lat_p95_ms": float(c16.lat_ms_p95.mean()) if len(c16) else float("nan"),
            "cost_per_1m_usd": gpu_price / (tok * 3600) * 1e6 if tok == tok and tok > 0 else float("nan"),
            "load_errors": int(l.errors.sum()),
        }
        if name in evals:
            e = evals[name]
            r["quality_of_100"] = int(e.correct.sum())
            if "finish" in e.columns:
                r["truncated"] = int((e.finish == "length").sum())
        rows.append(r)
    df = pd.DataFrame(rows).set_index("config")
    if "baseline" in df.index:
        b = df.loc["baseline"]
        df["speedup_vs_baseline"] = df.tok_s / b.tok_s
        if "quality_of_100" in df.columns and pd.notna(b.get("quality_of_100")):
            df["quality_retention"] = df.quality_of_100 / b.quality_of_100
    return df


def category_table(evals):
    rows = {}
    for name, e in evals.items():
        rows[name] = {c: f"{int(g.correct.sum())}/{len(g)}" for c, g in e.groupby("category")}
    return pd.DataFrame(rows).T


def read_text(path):
    return open(path, encoding="utf-8").read() if os.path.exists(path) else None


def main():
    st.title("Inference Optimization Lab")
    st.caption("What does each LLM-serving optimization buy, and what does it cost in answer quality? "
               "Measured with vLLM on a single free T4 GPU.")

    results_dir = find_results_dir()
    if results_dir is None:
        st.info("No results found. The app needs files named `load_<config>.csv` (and `eval_<config>.csv`) "
                "in one of the folders below. Run the notebook first, or point RESULTS_DIR at your results folder.")
        st.write("Folders checked:")
        for p in candidate_dirs():
            n = len(glob.glob(os.path.join(p, "load_*.csv"))) if os.path.isdir(p) else None
            st.code(f"{p}   ->   " + ("folder not found" if n is None else f"{n} load_*.csv files"))
        return

    loads, evals = load_all(results_dir)

    gpu_price = st.sidebar.slider("GPU price ($/hr), an assumption", 0.0, 5.0, 0.5, 0.05)
    api_price = st.sidebar.slider("API price ($ per 1M tokens), an assumption", 0.0, 10.0, 1.0, 0.1)
    st.sidebar.caption("Prices are inputs you choose. They only affect the cost numbers, not speed or quality.")

    summary = build_summary(loads, evals, gpu_price)
    invalid = summary[summary.load_errors > 0].index.tolist()
    if invalid:
        st.warning(f"Load test had failed requests for: {', '.join(invalid)}. Their speed numbers are not valid.")

    tabs = st.tabs(["Overview", "Speed", "Quality", "Cost", "Router", "Method"])

    with tabs[0]:
        st.subheader("All configs (speed measured at concurrency 16)")
        st.dataframe(summary.round(2))
        findings = read_text(os.path.join(results_dir, "findings.md"))
        if findings:
            st.markdown(findings)
        else:
            st.info("Add your own conclusions in results/findings.md (for example 'which config when'). "
                    "They will show up here.")

    with tabs[1]:
        labels = {"tokens_per_sec": "Tokens/sec", "ttft_ms_p50": "Time to first token p50 (ms)",
                  "ttft_ms_p95": "Time to first token p95 (ms)", "lat_ms_p95": "Request latency p95 (ms)",
                  "lat_ms_p99": "Request latency p99 (ms)"}
        metric = st.selectbox("Metric", list(labels), format_func=lambda k: labels[k])
        chosen = st.multiselect("Configs", list(loads), default=list(loads))
        if chosen:
            chart = pd.DataFrame({n: loads[n].groupby("concurrency")[metric].mean() for n in chosen})
            st.line_chart(chart)
            st.caption("X axis: concurrency (requests in flight at once). Each point is the mean of 3 repeats.")

    with tabs[2]:
        if evals:
            st.subheader("Correct answers out of 100")
            if "quality_of_100" in summary.columns:
                st.bar_chart(summary["quality_of_100"].dropna())
            st.subheader("By category")
            st.dataframe(category_table(evals))
            st.caption("Each question is 1 point. With only 100 questions (30 to 40 per category), a gap of a few "
                       "questions is within noise.")
        else:
            st.info("No eval_*.csv files found.")

    with tabs[3]:
        monthly_gpu = gpu_price * HOURS_PER_MONTH
        c1, c2 = st.columns(2)
        c1.metric("Always-on GPU, $/month", f"{monthly_gpu:,.0f}")
        if api_price > 0:
            c2.metric("Break-even vs API, M tokens/month", f"{monthly_gpu / api_price:,.0f}")
        st.caption("Cost per 1M tokens = GPU $/hr / (tokens/sec x 3600) x 1,000,000, assuming the GPU is fully used. "
                   "Below the break-even volume the API is cheaper.")
        st.bar_chart(summary["cost_per_1m_usd"].dropna())

    with tabs[4]:
        sweep_path = os.path.join(results_dir, "router_sweep.csv")
        if os.path.exists(sweep_path):
            sw = pd.read_csv(sweep_path)
            st.line_chart(sw.set_index("cost_per_1m")["quality_of_100"])
            st.dataframe(sw.round(3))
            st.caption("A TF-IDF + logistic-regression classifier predicts whether the small model will answer "
                       "correctly. Predictions are out-of-fold (5-fold cross-validation).")
        else:
            st.info("No router_sweep.csv yet. Run the router section of the notebook.")

    with tabs[5]:
        st.markdown("""
**Method**
- One change per config; same 100-question eval (40 math, 30 knowledge, 30 JSON format), scored by code.
- Eval: temperature 0, 768 max tokens. Load test: concurrency 1, 4, 16, 32; 48 requests per run; 3 repeats; 128 forced output tokens; warm-up discarded.
- Server settings shared by all configs: FP16 activations, max model length 2048, GPU memory utilization 0.85, eager mode.

**Limitations**
- Small eval set, one GPU type, free-tier hardware that can be shared or throttled.
- Eager mode lowers absolute speeds for every config.
- Steady synthetic load, not real traffic.
- Cost uses an assumed GPU price and full utilization.
""")
        env = read_text(os.path.join(results_dir, "env.txt"))
        if env:
            st.code(env)
        for p in sorted(glob.glob(os.path.join(results_dir, "mem_*.txt"))):
            with st.expander("Server log lines: " + os.path.basename(p)[4:-4]):
                st.code(read_text(p) or "(no matching lines)")


main()

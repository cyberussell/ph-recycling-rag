"""Streamlit frontend — a thin client over the FastAPI backend.

Deliberately thin: the RAG pipeline (routing, hybrid search, rerank,
agentic loop, model routing, generation) is the portfolio centerpiece, not
the UI. This app's only real job is rendering the answer and making the
pipeline's internals visible via the "How I found this" panel, which is what
actually demonstrates the hybrid-search + rerank + agentic-loop work to a
reviewer — not just the final answer text.
"""

import os
import subprocess
import sys
import time

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

EXAMPLE_QUESTIONS = [
    "Can I recycle a Tetra Pak?",
    "What is the penalty for littering?",
    "Magkano ang multa sa pagkalat ng basura?",
    "How do I segregate my household waste?",
]


@st.cache_resource
def _ensure_backend_running() -> bool:
    """Hugging Face Spaces' Streamlit SDK runs a single process, so when
    API_URL still points at localhost (i.e. no external API was configured)
    this spawns the FastAPI backend as a background subprocess exactly once
    per container lifetime. Local dev where you've started uvicorn yourself
    is unaffected — this only fires if nothing is already listening."""
    def reachable() -> bool:
        try:
            return requests.get(f"{API_URL}/health", timeout=2).ok
        except requests.RequestException:
            return False

    if reachable():
        return True
    if "localhost" not in API_URL and "127.0.0.1" not in API_URL:
        return False  # pointed at a real external API that just isn't up yet

    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
    )
    for _ in range(30):
        if reachable():
            return True
        time.sleep(2)
    return False


st.set_page_config(page_title="PH Recycling Assistant", page_icon="♻️")
_ensure_backend_running()
st.title("♻️ PH Recycling Assistant")
st.caption(
    "Ask about Philippine solid waste segregation, recycling, and RA 9003 rules. "
    "Not legal advice — verify with official DENR/EMB/LGU sources."
)

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

st.write("Try:")
cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, q in zip(cols, EXAMPLE_QUESTIONS):
    if col.button(q, use_container_width=True):
        # Must be set before the text_input below is (re)instantiated — a
        # keyed widget's own session_state entry wins over a `value=` kwarg
        # on reruns, so `value=` alone can't be used to push a click into it.
        st.session_state.question_input = q

question = st.text_input("Your question", key="question_input")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Classifying, searching, and generating an answer..."):
        result = None
        try:
            # 240s: on free CPU-only hosting the agentic loop's worst case
            # (reformulate-and-retry, each attempt paying full embed+rerank
            # cost) can genuinely take a couple minutes — 120s was hit for
            # real on Hugging Face's cpu-basic tier and errored out a
            # request that would have succeeded. Real latency reduction
            # lives in hybrid_search.py's FUSION_CANDIDATES; this is just
            # the safety margin so a slow-but-working answer isn't treated
            # as a failure.
            resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=240)
            if resp.status_code == 429:
                st.warning(resp.json().get("detail", "Demo request limit reached — please try again tomorrow."))
            elif resp.status_code == 503:
                st.info(resp.json().get("detail", "Still warming up — please retry in a moment."))
            else:
                resp.raise_for_status()
                result = resp.json()
        except requests.RequestException as e:
            st.error(f"Couldn't reach the API at {API_URL}: {e}")

    if result:
        st.markdown(result["answer"])

        if result["sources"]:
            st.subheader("Sources")
            for s in result["sources"]:
                st.markdown(f"- [{s['title']}]({s['url']})")

        with st.expander("🔍 How I found this"):
            debug = result["debug"]
            st.write(f"**Classified as:** `{debug['intent']}` — {debug['intent_reason']}")

            if debug["out_of_scope"]:
                st.write("Out of scope — redirected before any retrieval or generation.")
            else:
                st.write(f"**Model used:** `{debug['model']}`" + (" (escalated)" if debug["escalated"] else ""))
                if debug["escalated"]:
                    st.caption(f"Escalation reason: {debug['escalate_reason']}")

                st.write("**Retrieval attempts:**")
                for i, attempt in enumerate(debug["attempts"], start=1):
                    st.write(f"{i}. `{attempt['query']}`")
                    st.caption(
                        f"sufficient={attempt['sufficient']} — {attempt['reason']}"
                    )
                if debug["reformulated_query"]:
                    st.info(f"Reformulated query: {debug['reformulated_query']}")

                st.write("**Retrieved chunks (post-rerank):**")
                for c in debug["chunks"]:
                    loc = c["section_id"] or f"p.{c['page_number']}"
                    st.caption(
                        f"rerank={c['rerank_score']:.3f}  {c['source_title']} {loc}"
                        + (f" — {c['section_title']}" if c["section_title"] else "")
                    )

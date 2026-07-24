"""Simple chat UI for the insurance RAG system.

Run with: streamlit run streamlit_app.py
"""
import streamlit as st

from rag import config, generation, metadata, retrieval

st.set_page_config(page_title="Insurance RAG - Bajaj Allianz Motor", page_icon=":motorcycle:")
st.title("Insurance RAG Assistant")
st.caption("Ask about Bajaj Allianz / Bajaj General Insurance two-wheeler motor policies")

products = sorted({v["product"] for v in metadata.DOCUMENT_METADATA.values()})
doc_types = sorted({v["document_type"] for v in metadata.DOCUMENT_METADATA.values()})

with st.sidebar:
    st.header("Filters")
    product = st.selectbox("Product", ["(any)"] + products)
    doc_type = st.selectbox("Document type", ["(any)"] + doc_types)
    expand = st.checkbox("Expand to full section context", value=False)
    top_k = st.slider("Chunks to retrieve", 3, 15, config.TOP_K)
    if st.button("Clear chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about these motor policies...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    filters = {}
    if product != "(any)":
        filters["product"] = product
    if doc_type != "(any)":
        filters["document_type"] = doc_type

    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            chunks = retrieval.hybrid_search(
                query, top_k=top_k, filters=filters or None, expand_context=expand
            )
            result = generation.generate_answer(query, chunks)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander(f"Sources ({len(result['sources'])})"):
                for src, page in result["sources"]:
                    st.markdown(f"- **{src}**, p.{page}")
        if result["backend"]:
            st.caption(f"Answered via {result['backend']}")

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

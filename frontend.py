import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

# Import the actual async functions from your pipeline
from rag_pipeline import ingest_document, query_document

load_dotenv()

# Helper to run the async pipeline functions within Streamlit's sync environment
def run_async(coro):
    return asyncio.run(coro)

# --- Page Config ---
st.set_page_config(page_title="Doc-Lense Analyst", page_icon="📊")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Doc-Lense: Financial AI Analyst")

# --- Session State ---
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None

# --- Sidebar: Document Ingestion ---
with st.sidebar:
    st.header("📁 Document Upload")
    # Your pipeline uses 'document_id' to isolate data, so we use the filename
    uploaded_file = st.file_uploader("Upload a Financial PDF", type=["pdf"])
    
    if uploaded_file:
        # We use the filename (cleaned) as the ID for ChromaDB
        doc_id = uploaded_file.name.replace(" ", "_")
        
        if st.button("Analyze Document"):
            with st.spinner("Reading and indexing document..."):
                try:
                    pdf_bytes = uploaded_file.read()
                    # Call the async ingest function
                    run_async(ingest_document(pdf_bytes, doc_id))
                    
                    st.session_state.doc_id = doc_id
                    st.success("✅ Document Ready!")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

# --- Main Interface ---
if st.session_state.doc_id:
    st.subheader(f"💬 Analyzing: {st.session_state.doc_id}")
    user_question = st.text_input("e.g., 'What was the total revenue in 2023?'")

    if st.button("Get Analysis"):
        if user_question:
            with st.spinner("Analyzing data across pages..."):
                try:
                    # Call the async query function
                    result = run_async(query_document(st.session_state.doc_id, user_question))
                    
                    st.markdown("### 🧠 Analysis Process")
                    for step in result.steps:
                        # Expanded for Chain of Thought, collapsed for final output usually looks best
                        is_output = (step.step == "OUTPUT")
                        with st.expander(f"Step: {step.step}", expanded=is_output):
                            st.write(step.content)
                    
                    if result.pages:
                        st.caption(f"**Sources:** Verified data found on pages {', '.join(map(str, result.pages))}")

                except Exception as e:
                    st.error(f"Analysis failed: {e}")
        else:
            st.warning("Please enter a question first.")
else:
    st.warning("👈 Please upload and 'Analyze' a document in the sidebar to start.")

st.divider()
st.caption("Powered by LangChain & OpenAI | Doc-Lense 2026")
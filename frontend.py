import streamlit as st
import os
from dotenv import load_dotenv

# Import your existing RAG functions from your pipeline script
from rag_pipeline import ingest, query

# Load local environment variables (for local testing)
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Doc-Lense Analyst", page_icon="📊")

# --- CSS for a cleaner look ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .stTextInput>div>div>input { border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Doc-Lense: Financial AI Analyst")
st.info("I am an expert AI financial document analyst. Upload your PDF to begin.")

# --- Session State Initialization ---
# This keeps the vector database in memory so you don't re-ingest on every click
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- Sidebar: Document Ingestion ---
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload a Financial PDF", type=["pdf"])
    
    if uploaded_file:
        if st.button("Analyze Document"):
            with st.spinner("Reading and indexing document..."):
                try:
                    # Convert UploadedFile to bytes
                    pdf_bytes = uploaded_file.read()
                    # Call your existing 'ingest' function from rag_pipeline.py
                    st.session_state.vector_db = ingest(pdf_bytes)
                    st.success("✅ Document Ready!")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

# --- Main Interface: The Analysis ---
if st.session_state.vector_db is not None:
    st.subheader("💬 Ask a Financial Question")
    user_question = st.text_input("e.g., 'What was the total revenue in 2023?'")

    if st.button("Get Analysis"):
        if user_question:
            with st.spinner("Analyzing data across pages..."):
                try:
                    # Call your existing 'query' function from rag_pipeline.py
                    # This returns your Pydantic 'QueryResponse' model
                    result = query(st.session_state.vector_db, user_question)
                    
                    # --- Display Chain of Thought ---
                    st.markdown("### 🧠 Analysis Process")
                    for step in result.steps:
                        with st.expander(f"Step: {step.step}", expanded=True):
                            st.write(step.content)
                    
                    # --- Display Final Answer ---
                    st.success("### 📢 Final Answer")
                    # Finding the 'OUTPUT' step from your model
                    final_output = next((s.content for s in result.steps if s.step == "OUTPUT"), "No output generated.")
                    st.write(final_output)

                    # --- Display Sources ---
                    if result.pages:
                        st.caption(f"**Sources:** Verified data found on pages {', '.join(map(str, result.pages))}")

                except Exception as e:
                    st.error(f"Analysis failed: {e}")
        else:
            st.warning("Please enter a question first.")
else:
    st.warning("👈 Please upload and 'Analyze' a document in the sidebar to start.")

# --- Footer ---
st.divider()
st.caption("Powered by LangChain & OpenAI | Doc-Lense 2026")
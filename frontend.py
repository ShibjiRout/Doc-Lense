import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

# Import ALL functions from your pipeline, including delete
from rag_pipeline import ingest_document, query_document, delete_document

load_dotenv()

# Helper to run async functions
def run_async(coro):
    return asyncio.run(coro)

# --- Page Config ---
st.set_page_config(page_title="Doc-Lense Analyst", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    /* Make the delete button red */
    [data-testid="stSidebar"] button[kind="primary"] { background-color: #ff4b4b; color: white; border: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Doc-Lense: Financial AI Analyst")

# --- Session State Initialization ---
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "messages" not in st.session_state:
    st.session_state.messages = [] # This holds our continuous chat history

# --- Sidebar: Case Management & Upload ---
with st.sidebar:
    st.header("📁 Case Management")
    
    # 1. Unique Case ID Input
    custom_case_id = st.text_input("Enter Unique Case ID", value="Case_001")
    uploaded_file = st.file_uploader("Upload a Financial PDF", type=["pdf"])
    
    if uploaded_file and st.button("Analyze Document"):
        with st.spinner("Reading and indexing document..."):
            try:
                pdf_bytes = uploaded_file.read()
                # Use the custom Case ID provided by the user
                run_async(ingest_document(pdf_bytes, custom_case_id))
                
                st.session_state.doc_id = custom_case_id
                # Clear chat history when a new document is uploaded
                st.session_state.messages = [] 
                st.success(f"✅ Indexed under Case ID: {custom_case_id}")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    # 2. The Delete Button
    if st.session_state.doc_id:
        st.divider()
        st.subheader("⚠️ Danger Zone")
        # type="primary" triggers the red CSS we wrote at the top
        if st.button("🗑️ Delete Case Data", type="primary"):
            with st.spinner("Deleting from database..."):
                success = run_async(delete_document(st.session_state.doc_id))
                if success:
                    st.session_state.doc_id = None
                    st.session_state.messages = []
                    st.success("Case completely deleted from database.")
                    st.rerun() # Refresh the page
                else:
                    st.error("Failed to delete the case.")

# --- Main Interface: Continuous Chat ---
if st.session_state.doc_id:
    st.subheader(f"💬 Active Case: {st.session_state.doc_id}")
    
    # Render existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                # If it's the AI, display the steps and the final answer
                for step in msg["steps"]:
                    is_output = (step.step == "OUTPUT")
                    with st.expander(f"Step: {step.step}", expanded=is_output):
                        safe_content = step.content.replace("$", r"\$")
                        st.write(safe_content)
                if msg["pages"]:
                    st.caption(f"**Sources:** Verified data found on pages {', '.join(map(str, msg['pages']))}")

    # 3. Continuous Chat Input Box (Sticks to the bottom of the screen)
    if prompt := st.chat_input("Ask a financial question about this case..."):
        
        # Display user message instantly
        with st.chat_message("user"):
            st.write(prompt)
        
        # Save user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                try:
                    result = run_async(query_document(st.session_state.doc_id, prompt))
                    
                    for step in result.steps:
                        is_output = (step.step == "OUTPUT")
                        with st.expander(f"Step: {step.step}", expanded=is_output):
                            safe_content = step.content.replace("$", r"\$")
                            st.write(safe_content)
                    
                    if result.pages:
                        st.caption(f"**Sources:** Verified data found on pages {', '.join(map(str, result.pages))}")
                    
                    # Save AI response to history so it stays on screen
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "steps": result.steps,
                        "pages": result.pages
                    })
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

else:
    st.info("👈 Please set a Case ID and upload a document to begin the analysis.")
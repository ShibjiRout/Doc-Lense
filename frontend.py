import streamlit as st
import requests

# ── Configuration ─────────────────────────────────────────────────────────────
# IMPORTANT: When running locally, use localhost. 
# Once your backend is on Render, change this to your Render backend URL!
BACKEND_URL = "http://localhost:8000" 
# Example for production: BACKEND_URL = "https://your-fastapi-backend.onrender.com"

st.title("📄 AI Financial Document Analyst")
st.write("Upload a financial PDF and ask questions about it!")

# ── Sidebar: File Upload ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Ingesting document..."):
                # Send the file to the FastAPI backend
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{BACKEND_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success("Document successfully processed and ready!")
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the backend server. Is it running?")

# ── Main Chat Interface ───────────────────────────────────────────────────────
st.header("2. Ask Questions")
question = st.text_input("Enter your finance-related question:")

if st.button("Submit Question"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing..."):
            # Send the question to the FastAPI backend
            payload = {"question": question}
            try:
                response = requests.post(f"{BACKEND_URL}/ask", data=payload)
                if response.status_code == 200:
                    data = response.json()
                    
                    if "error" in data:
                        st.error(data["error"])
                    else:
                        # Display the Chain of Thought steps
                        st.subheader("Analysis Process")
                        for step in data.get("steps", []):
                            st.write(f"**{step['step']}**: {step['content']}")
                        
                        # Display the source pages
                        if data.get("pages"):
                            st.info(f"📚 Sources found on pages: {', '.join(map(str, data['pages']))}")
                else:
                    st.error(f"Error answering question: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend server.")
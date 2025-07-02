import streamlit as st

st.set_page_config(page_title="DocChat Duel", layout="wide")

st.title("📄🤖 DocChat Duel")

st.markdown("Upload a PDF and ask questions about its content!")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

question = st.text_input("Ask a question about the document:")

if question:
    st.info(f"You asked: {question}")
    # Include the backend RAG logic

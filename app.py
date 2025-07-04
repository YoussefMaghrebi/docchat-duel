import streamlit as st
import fitz  # PyMuPDF
import re
from langchain_app.utils import chunk_text
from langchain_app.embedding_utils import load_embedding_model, embed_chunks
from langchain_app.search_utils import search_faiss
from langchain_app.llm_utils import load_llm_model, generate_answer, remove_near_duplicates

def clean_chunk(text):
    """
    Cleans the input text chunk from unwanted and unimportant characters and words

    Args:
        text (str): input text chunk to be processed
    
    Returns:
        cleaned_text (str) : processed text chunk

    """

    lines = text.split("\n")
    cleaned = []
    
    for line in lines:
        line = line.strip()
        
        # Remove common numbering formats (e.g., "1. ", "2.3. ", "IV. ", "Chapter 5", etc.)
        numbered_line = re.sub(r"^(?:\d+[\.\d]*|[IVXLCDM]+)\.?\s*", "", line, flags=re.IGNORECASE)

        # Normalize the line to lowercase for keyword checks
        lower_line = numbered_line.lower()
        
        # Skip lines that are too short or start with unwanted section titles
        if len(numbered_line) < 10:
            continue

        if any(
            lower_line.startswith(h) 
            for h in [
                "abstract", "keywords", "related work", "acknowledgement", "conclusion",
                "chapter", "section", "references", "table of contents", "contents"
            ]
        ):
            continue

        keywords = [str(i) for i in range(1, 101)]  # ['1', '2', ..., '100']

        if "chapter" in lower_line and any(word in lower_line for word in keywords):
            continue

        if numbered_line.replace(".", "").replace(",", "").isdigit():
            continue

        cleaned.append(line)
    
    cleaned_text = "\n".join(cleaned)
    return cleaned_text


st.set_page_config(page_title="DocChat Duel", layout="wide")

st.title("📄🤖 DocChat App")

st.markdown("Upload PDF documents and ask questions about their content!")

# upload PDF files 
uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

# parse full text from all PDF files 
full_text = ""

if uploaded_files:
    for file in uploaded_files:
        st.success(f"Uploaded: {file.name}")

        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                full_text += page.get_text()

    # chunk the text
    all_chunks = chunk_text(full_text, max_words=500, overlap=100)   

    # clean the chunks (preprocessing)
    all_chunks = [clean_chunk(chunk) for chunk in all_chunks if chunk.strip()]

    # embed the chunks
    embedding_model = load_embedding_model()
    embeddings = embed_chunks(all_chunks, embedding_model)

    # load the LLM model 
    #model_id = "tiiuae/falcon-rw-1b"
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer, model = load_llm_model(model_name= model_id)
    st.info("The provided documents are processed ✅")

    # receive the user query 
    question = st.text_input("Ask a question about the documents:")

    if question:
        st.info("Thinking about the question, please wait...")

        # perform vector search
        top_k = 1
        top_chunks = search_faiss(question, embeddings, all_chunks, embedding_model, k=top_k)
        chunk_texts = [chunk for _, chunk in top_chunks] 
        
        # generate response using the loaded LLM model
        response = generate_answer(chunk_texts, question, tokenizer, model)

        # post process the LLM response 
        answer = remove_near_duplicates(response)

        # output the final generated answer 
        st.markdown("Answer:")
        st.markdown(f"<div style='text-align: justify;'>{answer}</div>", unsafe_allow_html=True)


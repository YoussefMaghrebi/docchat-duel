import fitz  # PyMuPDF
import os
import nltk

def extract_text_from_pdf(file_path):
    """
    Extracts raw text from a single PDF file using PyMuPDF.
    Returns the full extracted text as a string.
    """
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Failed to parse {file_path}: {e}")
    return text

def load_all_pdfs_from_folder(folder_path):
    """
    Loads and extracts text from all PDF files in the given folder.
    Returns a dictionary: {filename: extracted_text}
    """
    all_texts = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            all_texts[filename] = extract_text_from_pdf(full_path)
    return all_texts

def chunk_text(text, max_words=500, overlap=100):
    """
    Splits a long text into overlapping chunks based on sentence boundaries.
    
    Args:
        text (str): The full text to split.
        max_words (int): Maximum number of words per chunk.
        overlap (int): Number of words to overlap between chunks.

    Returns:
        List of chunks (each chunk is a string)
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        # If adding this sentence would exceed the limit
        if current_word_count + word_count > max_words:
            chunks.append(" ".join(current_chunk))

            # Start new chunk with overlap from previous one
            if overlap > 0:
                overlap_words = " ".join(current_chunk).split()[-overlap:]
                current_chunk = overlap_words.copy()
                current_word_count = len(overlap_words)
            else:
                current_chunk = []
                current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += word_count

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

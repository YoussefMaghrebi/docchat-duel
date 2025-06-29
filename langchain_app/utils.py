import fitz  # PyMuPDF
import os

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

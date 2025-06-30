from langchain_app.utils import load_all_pdfs_from_folder, chunk_text
import nltk


# Download NLTK’s pre-trained sentence tokenizer model (punkt) if it wasn't downloaded already
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')


if __name__ == "__main__":
    folder = "docs"
    texts = load_all_pdfs_from_folder(folder)

    for filename, content in texts.items():
        print(f"\n===== {filename} =====")
        chunks = chunk_text(content, max_words=500, overlap=100)
        print(f"Generated {len(chunks)} chunks.\n")

        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} ---\n")
            print(chunk[:600])  # Print first 600 characters of each chunk

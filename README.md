# 📚 DocChat Duel

This project compares two document Q&A pipelines using different LLM frameworks. It parses PDFs, chunks the content, and prepares for embedding and retrieval using LangChain and alternative tooling.

## 🚀 Features

- ✅ PDF parsing and text extraction
- ✅ Sentence-aware text chunking using NLTK
- 📥 Embedding and vector search 
- 💬 LLM-based Q&A interface (coming next)
- 🧪 Model framework comparison (LangChain vs others)

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YoussefMaghrebi/docchat-duel.git
cd docchat-duel
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate          # On Windows
# OR
source .venv/bin/activate       # On macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
docchat-duel/
├── langchain_app/
│   └── utils.py               # PDF loading + chunking logic
├── tests/
│   └── test_chunker.py        # Chunking test with sample PDFs
│   └── test_pdf_parser.py     # PDF parsing test with sample PDFs
├── docs/
│   └── *.pdf                  # Add your test PDFs here
├── requirements.txt
├── .gitignore
└── README.md
```

---
## 🔠 NLTK Setup Notes (important)

This project uses NLTK's `punkt` tokenizer for sentence chunking.

If you run the chunking test (`python tests/test_chunker.py`) for the first time, it will automatically download the necessary NLTK data models, including:

- `punkt`
- `wordnet`
- `omw-1.4`
- `punkt_tab`

If the download is successful, the chunking will work correctly without errors.
Note: The file test_chunker.py must be run from the project root folder.

## Running Tests

All the following test scripts must be run **from the project root** folder to ensure proper imports.

### 📄 PDF Parser Test

The `test_pdf_parser.py` script in the `tests/` folder is designed to verify the PDF text extraction logic.  

- Loads all PDF files placed in the `docs/` folder.
- Extracts and prints samples of the raw text content from each PDF file.

Run the test script as so:

```bash
python test_pdf_parser.py
```

### 🧪 Run the Chunking Test

To verify the tokenizer is working fine, run the file `test_chunker.py` found in the `tests/` directory:

```bash
python test_chunker.py
```

This will:
- Load PDFs from the `docs/` folder
- Apply sentence-aware chunking
- Print the first few chunks and chunk count

---

## 🔮 Coming Next

- Embedding logic using OpenAI + FAISS
- LangChain vs alternative framework implementation
- Streamlit / web UI integration
- Model evaluation metrics

---

## 📄 License

MIT License
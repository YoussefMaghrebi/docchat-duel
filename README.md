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
│   └── embedding_utils.py      # Loading embedding model + Vector embedding logic
│   └── utils.py                # PDF loading + chunking logic
│   └── search_utils.py         # Vector search logic
├── tests/
│   └── test_chunker.py         # Chunking test with sample PDFs
│   └── test_embedder.py        # Vector embedding test with sample chunks list
│   └── test_pdf_parser.py      # PDF parsing test with sample PDFs
│   └── test_pipeline.py        # Full pipeline test with sample PDFs
│   └── test_vector_search.py   # Vector search test using FAISS 
├── docs/
│   └── *.pdf                   # Add your test PDFs here
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

## 🧪 Running Tests

All the following test scripts must be run **from the project root** folder to ensure proper imports.

### PDF Parser Test

The `test_pdf_parser.py` script in the `tests/` folder is designed to verify the PDF text extraction logic.  

- Loads all PDF files placed in the `docs/` folder.
- Extracts and prints samples of the raw text content from each PDF file.

Run the test script as so:

```bash
python test_pdf_parser.py
```

### Run the Chunking Test

To verify the tokenizer is working fine, run the file `test_chunker.py` found in the `tests/` directory:

```bash
python test_chunker.py
```

This will:
- Load PDFs from the `docs/` folder
- Apply sentence-aware chunking
- Print the first few chunks and chunk count

### Run Embedding Test

To test the embedding generation, run the `test_embedder.py` script found inside `tests/`. This script uses the Hugging Face `sentence-transformers/all-MiniLM-L6-v2` model.

**Important:**  
The Hugging Face caching system tries to use symbolic links ("symlinks") on your disk to save space when storing model files. On Windows, creating symlinks requires either running Python with administrator rights or enabling Developer Mode.  
- If these are not set, the cache system will fall back to copying files instead of symlinking, which raises a warning and may use more disk space.  
- This warning can be safely ignored, but you can avoid it by running your terminal or VSCode as administrator or enabling Developer Mode in Windows settings.

The model and related tokenizer/config files are downloaded only once and cached locally (usually under `C:\Users\<username>\.cache\huggingface\hub`), so subsequent runs load from cache and do not download again.

### Run Full Pipeline Test

The script `test_pipeline.py` tests the **entire pipeline** from PDF loading → text chunking → embedding generation.

This helps verify that all core components work well together and that model dependencies are downloaded and cached correctly.

```bash
python test_pipeline.py
```

### Run the Vector Search Test

The `test_vector_search.py` script in the `tests/` folder is designed to verify the vector search logic.  

- Chunks and embeds a sample text paragraph.
- Performs vector search based on a sample query and retrieves most relevant chunks.

Run the test script as so:

```bash
python test_vector_search.py
```

---

## 🔮 Coming Next

- LangChain vs alternative framework implementation
- Streamlit / web UI integration
- Model evaluation metrics

---

## 📄 License

MIT License
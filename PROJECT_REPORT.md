# 📚 Project Design Report: DocChat-Duel

## 🎯 Project Overview

**DocChat-Duel** is a lightweight, end-to-end prototype for report and document-based Q&A using Large Language Models (LLMs). Given one or more PDFs, the system extracts their content, chunks it semantically, embeds the chunks, and stores them for retrieval when answering user questions.

The goal is to:
- ✅ Build a modular and clean pipeline using open-source tools
- ✅ Compare LLM orchestration frameworks (LangChain vs others)
- ✅ Avoid paid APIs to enable free experimentation and sharing

---

## 🧱 System Components and Rationale

### 1. **PDF Parsing**

📂 **Component**: `load_all_pdfs_from_folder()` using `PyMuPDF` (`fitz`)  
📌 **Purpose**: Load and extract raw text from one or more PDF files

#### ✅ Why `PyMuPDF`?
| Option         | Pros                                       | Cons                        |
|----------------|--------------------------------------------|-----------------------------|
| `PyMuPDF`      | Fast, reliable text extraction, good layout | Slightly noisy with scanned content |
| `pdfplumber`   | Accurate layout-aware parsing               | Slower                      |
| `pdfminer`     | Low-level access, fine-grained control      | Complex API, Not ideal for quick pipelines        |

**Decision**: We chose `PyMuPDF` due to its speed, simplicity, and reliable parsing for structured text. Our use case is focused on analyzing **research papers and reports**, which are usually well-formatted and text-based — making `PyMuPDF` a strong fit.

---

### 2. **Text Chunking**

📂 **Component**: `chunk_text()`  
📌 **Purpose**: Split long texts into overlapping, sentence-aware chunks

#### ✅ Why `nltk.sent_tokenize()`?
| Option                  | Sentence Quality | Setup     | Flexibility   |
|-------------------------|------------------|-----------|----------------|
| `nltk.sent_tokenize()`  | ✅ Very Good      | Medium (requires punkt) | Good |
| `re.split()` (regex)    | ❌ Limited        | ✅ Easiest | Low            |
| `SpaCy`                 | ✅ Very Good      | ❌ Heavy   | High (NER, POS etc.) |
| `LangChain Splitters`   | ❌ No sentence awareness | ✅ LLM-friendly | Great for token-aware use |

**Decision**: We chose **NLTK** for sentence-aware chunking to preserve meaning and coherence during retrieval. It offers a high-quality split for academic and structured documents with minimal weight.

---

### 3. **Embedding Generation**

📂 **Component**: `embed_chunks()` using **Hugging Face (HF) `sentence-transformers`**  
📌 **Purpose**: Convert text chunks into semantic vector embeddings

#### ✅ Why Hugging Face (`all-MiniLM-L6-v2`)?

| Option                            | Cost     | Accuracy  | Speed   | Local Use | Notes                          |
|----------------------------------|----------|-----------|---------|-----------|-------------------------------|
| OpenAI `text-embedding-ada-002` | ❌ Paid   | ✅ Very High | ✅ Fast  | ❌ No      | State-of-the-art, but not free |
| HF `all-MiniLM-L6-v2`           | ✅ Free  | ✅ High    | ✅ Fast | ✅ Yes     | Excellent for RAG and offline |
| Instructor-XL (HF)              | ✅ Free  | 🔥 Great   | ❌ Slow | ✅ Yes     | Instruction-tuned, heavier     |

**Decision**: We opted for `all-MiniLM-L6-v2` via `sentence-transformers`:
- Fully offline and free
- Light and fast enough for real-time
- Proven to perform well in semantic search tasks

---

## 🧠 Business + Engineering Alignment

| Need                         | How It's Met                                                   |
|------------------------------|----------------------------------------------------------------|
| ✅ Local + Free stack        | All selected libraries work offline and are open-source        |
| ✅ Semantically accurate     | Sentence-based chunking + dense embedding preserves meaning     |
| ✅ Scalable pipeline         | Easy to expand with LangChain or LlamaIndex later              |
| ✅ Explainability            | Each decision documented and justified for maintainability     |
| ✅ Evaluation-ready          | Designed to support head-to-head comparisons of LLM frameworks |

---

## 🗺️ What's Next

| Module            | Description                                    |
|-------------------|------------------------------------------------|
| 🔜 Embedding logic| Implement `embed_chunks()` using HuggingFace   |
| 🔜 Vector store   | Use FAISS to index and retrieve relevant chunks|
| 🔜 LLM Response   | Connect to an LLM (e.g., OpenAI or LLaMA)      |
| 🔜 Frontend/API   | Optional UI to showcase Q&A from PDF input     |
| 🔜 Framework Duel | Compare LangChain vs. another orchestrator     |

---

## ✅ Conclusion

This project is an intentionally focused, extensible base for document Q&A and retrieval-augmented generation (RAG). Each decision reflects real-world constraints (cost, performance, reliability) and demonstrates a thoughtful engineering approach to solving LLM-backed search and question-answering (Q&A).
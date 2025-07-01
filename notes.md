# 📘 Current Ideas & Enhancements for DocChat-Duel

This document serves as a structured reference for future enhancements and design decisions.

---

#### ✅ **Current Architecture Summary**

* 🔍 **PDF Parsing:** Extract raw text from uploaded PDFs.
* ✂️ **Chunking:** Sentence-aware, overlapping chunks (via `nltk.sent_tokenize()`).
* 🧠 **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, 384-dim vectors.
* 🗃️ **Vector Search:** In-memory **FAISS IndexFlatIP** with cosine similarity.
* 💬 **Query Matching:** User query → embedding → top-k relevant chunks.

---

### 🧠 Future Enhancements (Advanced Retrieval)

#### 1. **Multi-Vector Query Representations**

| Method          | Idea                                             | Use Case                         |
| --------------- | ------------------------------------------------ | -------------------------------- |
| ColBERT         | Each token gets an embedding, then MaxSim scores | Better for long, complex queries |
| DensePhrases    | Phrase-level indexing and retrieval              | QA over large corpora            |
| Query Splitting | Manually split user query into 2–3 parts         | Paragraph-level questions        |

---

#### 2. **Hybrid Search (Dense + Sparse)**

Combine:

* **Dense Vectors** (semantic)
* **BM25 / TF-IDF** (keyword)

Frameworks: `Haystack`, `LlamaIndex`, `Weaviate`.

Use case: Technical documents or code bases with *domain-specific keywords*.

---

#### 3. **Re-Ranking with Cross-Encoder**

After FAISS retrieves top-10 chunks:

* Use **cross-encoder** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to **re-rank results** based on the full context of question + chunk.

Gives higher accuracy.

---

#### 4. **Memory Persistence & Index Caching**

Right now:

* FAISS index is rebuilt each time

Later:

* Save FAISS index: `index.save("docs.index")`
* Load later with: `faiss.read_index(...)`
* Improve loading time when working with **static corpora**

---

#### 5. **Support for LangChain or LlamaIndex**

If we want:

* Chat-based interface
* Integration with agents / tools / retrievers

Then:

* `LangChain` (modular, stable)
* `LlamaIndex` (faster prototyping, structured node-based abstraction)

---

#### 6. **UI Frontend Ideas**

* Upload documents
* Show search hits as context
* Highlight matches in document
* Chat history

Frontend Tools:

* Streamlit
* Gradio
* Next.js + FastAPI backend

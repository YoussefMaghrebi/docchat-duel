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

#### 5. ***Prompt Engineering & Token Limit Strategies***

As we continue to improve the response quality of the RAG pipeline, one promising direction is advanced **prompt engineering**. The current prompt format is relatively generic and sometimes fails to guide the model to avoid repetition or follow specific answer formats. Future work may include:

* Adding explicit **examples (few-shot prompts)** to guide the model toward desired answer structures.
* Experimenting with **system-level instructions** (e.g., role-based prompts or constraints).
* Dynamically adapting prompts based on query type (e.g., question, instruction, summarization).

Another limitation arises from the **token capacity** of our current language model (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`), which has a maximum sequence length of **2048 tokens**. This constraint affects both the prompt and the generated output. To work around this, we may explore:

* **Summarizing or compressing context chunks** before injecting them into the prompt.
* Using **hierarchical RAG**, where retrieved chunks are first summarized, then passed as distilled context.
* **Chunk ranking and filtering**, selecting only the most informative sentences within a chunk.
* Upgrading to models with larger token contexts (e.g., 2048+ tokens) when local resources allow.
* Implementing **context window sliding** to split the full query context into multiple windows and aggregate their outputs.

Token budget management and prompt design are both crucial levers for improving answer quality without needing to fine-tune or upscale the model. As such, both will be areas of active exploration in future iterations of the project.

---

#### 6. **Support for LangChain or LlamaIndex**

If we want:

* Chat-based interface
* Integration with agents / tools / retrievers

Then:

* `LangChain` (modular, stable)
* `LlamaIndex` (faster prototyping, structured node-based abstraction)

---

#### 7. **UI Frontend Ideas**

* Upload documents
* Show search hits as context
* Highlight matches in document
* Chat history

Frontend Tools:

* Streamlit
* Gradio
* Next.js + FastAPI backend

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
| HF `Instructor-XL`              | ✅ Free  | 🔥 Great   | ❌ Slow | ✅ Yes     | Instruction-tuned, heavier     |

**Decision**: We opted for `all-MiniLM-L6-v2` via `sentence-transformers`:
- Fully offline and free
- Light and fast enough for real-time
- Proven to perform well in semantic search tasks

### 4.  **Vector Indexing & Search** 

📂 **Component**: `search_faiss()` using **FAISS**  
📌 **Purpose**: Find the most semantically similar entries to a given query by comparing embedding vectors using a similarity metric (e.g. cosine similarity)

#### ✅ Why We Chose FAISS for This Project

| Feature                      | **FAISS**                           | **Annoy**                       | **HNSWlib**                    | **Pinecone**                      | **Chroma**                      | **Qdrant**                        |
|------------------------------|-------------------------------------|----------------------------------|--------------------------------|-----------------------------------|----------------------------------|-----------------------------------|
| 🧠 Type                      | In-memory w/ optional persistence   | In-memory (disk-saveable)        | In-memory (disk-saveable)      | Hosted vector DB (cloud)          | Lightweight local vector DB      | Full-featured vector DB           |
| 💾 Persistence               | ✅ Via `write_index()`              | ✅ Save to disk as `.ann`        | ✅ Save/load with binary        | ✅ Fully persistent                | ✅ Persistent                     | ✅ Persistent                      |
| 📐 Algorithm                 | Flat / IVF / HNSW (customizable)   | Random Projection Forests        | Hierarchical Navigable Small World | Proprietary                        | HNSW + LLM aware                | HNSW / Scalar Quantization        |
| 🧪 Accuracy                  | ✅ Very High (esp. Flat/HNSW)       | ⚠️ Lower than FAISS/HNSWlib      | ✅ Very High                    | ✅ High                            | ✅ High (for prototyping)        | ✅ High                            |
| ⚡ Speed                     | 🚀 Fastest on large queries         | ⚡ Fast (but limited tuning)      | ⚡ Fast and scalable            | ⚡ Cloud-scaled                    | ⚡ Good for RAG / small apps     | ⚡ Cloud/local hybrid              |
| 🔌 Ease of Use              | ✅ Easy (Python bindings)           | ✅ Very easy                     | ✅ Simple API                   | ✅ High-level API                  | ✅ Plug-and-play in LangChain    | ✅ REST + Python client            |
| 🔧 Distance Metrics          | Cosine, L2, IP (customizable)       | Cosine, Angular, Euclidean       | Cosine, L2                      | Cosine/IP (abstracted)            | Cosine (default)                | Cosine, dot-product, custom       |
| 🌐 Deployment               | Local                               | Local                            | Local                           | Cloud                             | Local                            | Local / Cloud                     |
| 📦 Size & Setup             | Medium (single dependency)          | Very small                       | Small                           | External infra                    | Light                            | Medium-heavy                      |
| 🧪 Best Use Case            | Research, RAG, scalable LLM tools   | Quick protos w/ static data      | Real-time, scalable search      | Scalable production RAG           | Lightweight dev tools            | High-scale semantic apps          |

Given our project goals and environment:

- 📄 **Our dataset is dynamic**: documents change per user session. We don't need long-term storage or persistence.
- 💡 **Simplicity**: FAISS can be used *in-memory*, avoids setting up external databases.
- 🚀 **Speed and Accuracy**: It supports **cosine similarity**, **inner product**, and **L2 distance**, with high accuracy using flat or HNSW indexing.
- 🛠️ **Flexibility**: Should we grow the dataset or add persistence later, FAISS scales with advanced index types (IVF, PQ, HNSW).
- 🤖 **LLM Compatibility**: Hugging Face + FAISS are often used together in open-source LLM and RAG pipelines.

While **Annoy** and **HNSWlib** are fast and simple, FAISS provides:
- Better **distance metric support**
- Larger **community and integrations**
- More advanced **indexing strategies** if needed in future

**Decision**: Therefore, we chose **FAISS** since it strikes the best balance between prototyping speed and production-readiness for our case.

### 5.  **Retrieval-Augmented Generation (RAG)** 

📂 **Component**: `generate_answer()` using **Falcon-RW-1B**  
📌 **Purpose**: Pass the retrieved chunks to a Large Language Model (LLM) to generate a final answer

#### ✅ Why We Chose Falcon-RW-1B

We evaluated several open-source language models for use in our RAG pipeline. Below is a comparison of key models that can be deployed locally without API costs:

| Model                          | Size       | Instruction-Tuned | Hardware Requirements      | Inference Speed   | Strengths                                                 | Best Use Case                                   | Cons                                                                 |
|-------------------------------|------------|-------------------|----------------------------|-------------------|-----------------------------------------------------------|------------------------------------------------|----------------------------------------------------------------------|
| **Mistral-7B-Instruct**        | ~7B        | ✅                | 24GB+ GPU recommended      | Moderate          | High-quality instruction responses, rich language ability | Production-quality answers with strong hardware | Heavy model, slow inference on local machines                       |
| **tiiuae/Falcon-RW-1B**        | ~1B        | ✅                | Low to Mid (4–6GB GPU)     | Very Fast         | Fast, lightweight, suitable for RAG prototyping           | Best tradeoff for local development & testing   | May produce shorter, simpler responses due to smaller capacity     |
| **google/flan-t5-base**        | ~400M      | ✅                | Very Low (CPU-compatible)  | Extremely Fast    | Fast inference, excels in structured QA & summarization   | Low-resource environments & quick inference     | Seq2seq model: doesn't handle long context as effectively in RAG   |

We selected **Falcon-RW-1B** as the first model for integration because it offers the **best trade-off between performance, speed, and hardware requirements**. It is instruction-tuned, fast, Hugging Face-hosted (easy to integrate) and works well on modest GPUs or CPUs, making it ideal for:

- Prototyping our document question answering system.
- Testing embeddings and vector search integration.
- Ensuring everything works end-to-end before scaling up to larger models.

Once the full system is validated, we can later upgrade to larger models like **Mistral-7B** for improved response quality.

### 6.  **Decoding Strategy Tuning**

To improve the quality of responses generated by the LLM, we experimented with several decoding strategies by adjusting the `temperature`, `top_k`, and `top_p` sampling parameters during text generation. We observed how these settings impacted fluency, relevance, and repetition in the answers returned by our Retrieval-Augmented Generation (RAG) pipeline.

We also introduced a **post-processing step** that removes **near-duplicate lines** to address repetitive outputs — particularly useful when decoding was less controlled.

---

#### 🔬 Experimental Comparisons

##### **1. Deterministic Generation (No Sampling)**

* **Settings**: `do_sample=False` (greedy decoding)
* **Observation**: Severe repetition, especially of phrases like "Speech Recognition", despite some relevant items.

```text
Natural Language Processing is used for the following:
1. Sentiment analysis
2. Machine Translation
3. Text Extraction
4. Text Mining
5. Speech Recognition
6. Speech Synthesis
7. Speech Recognition
8. Speech Recognition
...
24. Speech
```

**✅ Pros**: Predictable output
**❌ Cons**: Lots of repetition, limited diversity

---

##### **2. Sampling Enabled (top\_k=50, top\_p=0.9, temperature=0.7)** ✅ *Best Fit*

```text
Natural Language Processing is used for a variety of applications, including:
• Text extraction
• Sentiment analysis
• Speech recognition
• Machine translation
• Text summarization
• Natural language processing is used to process natural language.
5.
```

**✅ Pros**: Balanced diversity and fluency, no severe repetition
**❌ Cons**: Occasional irrelevant or malformed items (e.g., "5.")

---

##### **3. Slightly Lower Temperature (top\_k=50, top\_p=0.9, temperature=0.65)**

```text
• NLP is used for natural language processing. 
• It is used to convert human language into machine-understandable form. 
• It is used to analyze, understand, and process natural language. 
• NLP is used in speech recognition, natural language processing, automatic translation, and more.
...
```

**✅ Pros**: More structured
**❌ Cons**: Redundant phrases ("used in natural language processing...")

---

##### **4. Lower top\_p and temperature (top\_k=50, top\_p=0.85, temperature=0.6)**

```text
Natural language processing is used for a number of things. Some of these are:
- Translating speech to text.
- Translating text to speech.
- Extracting information from text.
- Searching for information in text.
- Finding information in a document.
- Finding information in a text.
...
```

**✅ Pros**: More readable, task-oriented
**❌ Cons**: Still suffers from phrase repetition ("Finding information in...")

---

##### **5. Very Low Temperature (top\_k=50, top\_p=0.9, temperature=0.5)**

```text
Natural language processing is used for the extraction of information from text, speech, and other sources.
It is used to extract information from text, speech, and other sources.
...
```

**✅ Pros**: Consistent topic
**❌ Cons**: Extreme redundancy and robotic tone

---

#### 🧹 Post-Processing for Repetition

In responses where repetition degraded the output quality, we implemented a **near-duplicate removal function**. This simple yet effective post-processing step significantly improved clarity, especially for lower-temperature generations or fully greedy decoding.

---

#### ✅ Best Trade-Off Setting

After empirical testing, the setting below provided the **most balanced result** in terms of fluency, specificity, and non-redundancy:

```python
top_k = 50  
top_p = 0.9  
temperature = 0.7  
do_sample = True
```

This configuration allows creative but not chaotic outputs and significantly reduces repeated phrases without over-randomizing the response.

---

## 🧠 Business + Engineering Alignment

| Need                         | How It's Met                                                   |
|------------------------------|----------------------------------------------------------------|
| ✅ Local + Free stack        | All selected libraries work offline and are open-source        |
| ✅ Semantically accurate     | Sentence-based chunking + dense embedding preserves meaning    |
| ✅ Scalable pipeline         | Easy integration with LangChain or LlamaIndex                  |
| ✅ Explainability            | Each decision documented and justified for maintainability     |
| ✅ Evaluation-ready          | Designed to support head-to-head comparisons of LLM frameworks |

---

## 🗺️ What's Next

| Module            | Description                                    |
|-------------------|------------------------------------------------|    
| 🔜 Frontend/API   | More UI features for a more user-friendly experience     |
| 🔜 Framework Duel | Compare LangChain vs. another orchestrator     |

---

## ✅ Conclusion

This project is an intentionally focused, extensible base for document Q&A and retrieval-augmented generation (RAG). Each decision reflects real-world constraints (cost, performance, reliability) and demonstrates a thoughtful engineering approach to solving LLM-backed search and question-answering (Q&A).
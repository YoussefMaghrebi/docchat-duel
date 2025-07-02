from langchain_app.utils import load_all_pdfs_from_folder, chunk_text
from langchain_app.embedding_utils import load_embedding_model, embed_chunks
from langchain_app.search_utils import search_faiss
from langchain_app.llm_utils import load_llm_model, generate_answer, remove_near_duplicates

# Step 1: Load and chunk PDFs
folder = "docs"
texts = load_all_pdfs_from_folder(folder)
all_chunks = []

for content in texts.values():
    chunks = chunk_text(content, max_words=500, overlap=100)
    all_chunks.extend(chunks)

# Step 2: Embed the chunks
embedding_model = load_embedding_model()
embeddings = embed_chunks(all_chunks, embedding_model)

# Step 3: Ask a query
query = "What is Natural Language Processing used for?"
top_k = 1
top_chunks = search_faiss(query, embeddings, all_chunks, embedding_model, k=top_k)
chunk_texts = [chunk for _, chunk in top_chunks] 

# Step 4: Load Falcon LLM model
model_id = "tiiuae/falcon-rw-1b"
tokenizer, model = load_llm_model(model_name= model_id)

# calculation of used nb of tokens
for i, chunk in enumerate(chunk_texts):
    chunk_tokens = tokenizer.encode(chunk, add_special_tokens=False)
    print(f"Chunk {i+1} tokens: {len(chunk_tokens)}")

prefix = (
    "You are an expert assistant. Carefully read the context below and answer the question fully.\n\n"
    "Context:\n"
    f"{'\n---\n'.join(chunk_texts)}\n\n"
    "Question:\n"
    f"{query}\n\n"
)

suffix = "Answer in clear, complete sentences (avoid lists unless explicitly requested), and avoid repetitions:"

prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

print(f"Prefix tokens: {len(prefix_tokens)}")
print(f"Suffix tokens: {len(suffix_tokens)}")

total_tokens = len(prefix_tokens) + len(suffix_tokens)
print(f"Total tokens for entire prompt: {total_tokens}")

# Step 5: Generate response using abstraction
response = generate_answer(chunk_texts, query, tokenizer, model)

# Count tokens in the generated answer
num_tokens = len(tokenizer.encode(response))
print(f"Generated {num_tokens} tokens for the answer.")

# print the final generated answer
print("\n===== RAG RESPONSE =====")
print(response)

# Step 6: Postprocessing - clean response from repetitions
print("\n===== Post-processed RESPONSE =====")
cleaned = remove_near_duplicates(response)
print(cleaned)
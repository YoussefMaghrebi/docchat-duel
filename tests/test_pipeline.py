from langchain_app.utils import load_all_pdfs_from_folder, chunk_text
from langchain_app.embedding_utils import load_embedding_model, embed_chunks

model = load_embedding_model()
folder = "docs"
texts = load_all_pdfs_from_folder(folder)

for filename, content in texts.items():
    print(f"\n===== {filename} =====")
    
    chunks = chunk_text(content, max_words=500, overlap=100)
    print(f"Generated {len(chunks)} chunks.\n")
    
    embeddings = embed_chunks(chunks)
    print(f"Generated {len(embeddings)} embeddings.\n")

    for i, emb in enumerate(embeddings[:2]):
        print(f"Embedding {i+1} shape: {emb.shape}")

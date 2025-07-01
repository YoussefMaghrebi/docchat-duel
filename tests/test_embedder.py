from langchain_app.embedding_utils import load_embedding_model, embed_chunks

# Load the embedding model 
model = load_embedding_model()

chunks = [
    "Natural language processing is a subfield of AI.",
    "It involves the interaction between computers and human language.",
    "Building AI systems is awesome!"
]

embeddings = embed_chunks(chunks, model)

for i, emb in enumerate(embeddings):
    print(f"Chunk {i+1} embedding shape: {emb.shape}")

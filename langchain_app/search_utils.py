import faiss
import numpy as np

def search_faiss(query, embeddings, chunks, model, k=3):
    """
    Search for the top-k most relevant chunks using FAISS vector similarity.
    
    Args:
        query (str): User query string.
        embeddings (List[np.array]): Precomputed document embeddings.
        chunks (List[str]): Original text chunks corresponding to embeddings.
        model: Embedding model e.g. SentenceTransformer model instance.
        k (int): Number of top results to return.

    Returns:
        List of tuples (score, chunk_text)
    """

    # Convert vector embeddings to numpy matrix
    embedding_matrix = np.array(embeddings).astype("float32")

    # Cosine similarity = Inner Product (IP) if both vectors are normalized,
    # Since FAISS doesn't have built-in cosine similarity, we normalize vectors and use IP:  
    
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])       # Inner product index initialized with vector size 
    faiss.normalize_L2(embedding_matrix)                       # Normalize document embedded vectors using L2 Norm
    index.add(embedding_matrix)

    # Embed and normalize query (for now the query is embedded into a single vector, in the future we might explore multi-vector query embedding)

    query_embedding = model.encode([query]).astype("float32")  # Embed query using the same model used to embed given PDF docs
    faiss.normalize_L2(query_embedding)                        # Normalize the query embedding vector using L2 Norm

    # Search
    scores, indices = index.search(query_embedding, k)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        results.append((score, chunks[idx]))

    return results

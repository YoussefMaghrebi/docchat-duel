from sentence_transformers import SentenceTransformer

def load_embedding_model():
    """
    Returns: Embedding model instance.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks, model):
    """
    Generate embeddings for a list of text chunks.

    Args:
        chunks (List[str]): List of text chunks.

    Returns:
        List[np.ndarray]: List of embedding vectors.
    """
    if not chunks:
        return []

    embeddings = model.encode(chunks, show_progress_bar=False)
    return embeddings

from sentence_transformers import SentenceTransformer

# Load the model globally so it's not reloaded on each call
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
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

from langchain_app.utils import chunk_text
from langchain_app.embedding_utils import load_embedding_model, embed_chunks
from langchain_app.search_utils import search_faiss

if __name__ == "__main__":
    text = """
    Artificial intelligence (AI) is the simulation of human intelligence in machines.
    These machines are programmed to think and learn. AI is being used in various fields such as healthcare, finance, and education.
    Natural language processing (NLP) is a key component of AI.
    It allows machines to understand and generate human language.
    """

    # Load the embedding model
    model = load_embedding_model()

    # Chunk and embed the text
    chunks = chunk_text(text, max_words=40, overlap=10)
    embeddings = embed_chunks(chunks, model)

    # Run vector search
    query = "What is NLP in AI?"
    results = search_faiss(query, embeddings, chunks, model, k=2)

    print("🔍 Top Results:")
    for idx, result in enumerate(results, 1):
        print(f"\n--- Result {idx} ---\n{result}")

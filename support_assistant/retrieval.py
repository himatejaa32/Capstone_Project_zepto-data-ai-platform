import chromadb

from sentence_transformers import SentenceTransformer


client = chromadb.PersistentClient(
    path="support_assistant/chroma_db"
)


collection = client.get_collection(
    name="zepto_policies"
)


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def retrieve(query, top_k=3):

    query_embedding = model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    documents = results["documents"][0]

    ids = results["ids"][0]

    return ids, documents
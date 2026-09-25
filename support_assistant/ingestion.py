import os

import chromadb

from sentence_transformers import SentenceTransformer


DOCS_DIR = "support_assistant/docs"


client = chromadb.PersistentClient(
    path="support_assistant/chroma_db"
)


collection = client.get_or_create_collection(
    name="zepto_policies"
)


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def load_documents():

    documents = []
    ids = []

    for filename in sorted(
        os.listdir(DOCS_DIR)
    ):

        if not filename.endswith(".txt"):
            continue

        path = os.path.join(
            DOCS_DIR,
            filename
        )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        documents.append(text)

        ids.append(
            filename.replace(
                ".txt",
                ""
            )
        )

    return ids, documents


def build_database():

    ids, documents = load_documents()

    embeddings = model.encode(
        documents
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

    print(
        f"Indexed {len(documents)} documents."
    )


if __name__ == "__main__":

    build_database()
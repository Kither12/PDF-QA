from langchain_core.documents import Document
from langchain_core.embeddings.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from uuid import uuid4
from langchain_cohere import CohereEmbeddings


class VectorDatabase:
    def __init__(self):
        client = QdrantClient(":memory:")

        client.create_collection(
            collection_name="pdf_qa",
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE,
            ),
        )

        embeddings = CohereEmbeddings(model="embed-english-v3.0")

        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name="pdf_qa",
            embedding=embeddings,
        )

    def add_documents(self, documents: list[Document]):
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids)

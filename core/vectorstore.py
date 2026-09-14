import chromadb

from core.models import Chunk

DEFAULT_PERSIST_DIR = "storage/chroma_db"


class VectorStore:
    def __init__(self, persist_directory: str = DEFAULT_PERSIST_DIR):
        self.client = chromadb.PersistentClient(path=persist_directory)

    def _collection(self, document_id: str):
        return self.client.get_or_create_collection(name=f"doc_{document_id}")

    def delete_document(self, document_id: str) -> None:
        self.client.delete_collection(name=f"doc_{document_id}")

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        collection = self._collection(chunks[0].document_id)
        collection.add(
            ids=[c.id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {"location_label": c.location_label, "source_type": c.source_type}
                for c in chunks
            ],
        )

    def get_all_chunks(self, document_id: str) -> list[dict]:
        collection = self._collection(document_id)
        results = collection.get()

        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            chunks.append(
                {
                    "id": chunk_id,
                    "text": results["documents"][i],
                    "location_label": metadata["location_label"],
                    "source_type": metadata["source_type"],
                }
            )
        return chunks

    def query(self, document_id: str, query_embedding: list[float], k: int = 5) -> list[dict]:
        collection = self._collection(document_id)
        results = collection.query(query_embeddings=[query_embedding], n_results=k)

        matches = []
        for i, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            matches.append(
                {
                    "id": chunk_id,
                    "text": results["documents"][0][i],
                    "location_label": metadata["location_label"],
                    "source_type": metadata["source_type"],
                    "distance": results["distances"][0][i],
                }
            )
        return matches

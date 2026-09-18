from core.models import Chunk
from core.vectorstore import VectorStore


def _chunk(id, document_id, location_label, text, embedding):
    return Chunk(
        id=id,
        document_id=document_id,
        source_type="pdf",
        location_label=location_label,
        text=text,
        embedding=embedding,
    )


def test_query_returns_the_closest_chunk_first(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    chunks = [
        _chunk("doc1-0", "doc1", "p.1", "payment due date", [1.0, 0.0, 0.0]),
        _chunk("doc1-1", "doc1", "p.2", "termination clause", [0.0, 1.0, 0.0]),
    ]
    store.add_chunks(chunks)

    results = store.query("doc1", query_embedding=[1.0, 0.0, 0.0], k=2)

    assert results[0]["id"] == "doc1-0"
    assert results[0]["location_label"] == "p.1"
    assert results[0]["text"] == "payment due date"
    assert results[0]["source_type"] == "pdf"


def test_query_respects_k_limit(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    chunks = [
        _chunk(f"doc1-{i}", "doc1", f"p.{i}", f"chunk {i}", [float(i), 0.0, 0.0])
        for i in range(5)
    ]
    store.add_chunks(chunks)

    results = store.query("doc1", query_embedding=[0.0, 0.0, 0.0], k=2)

    assert len(results) == 2


def test_documents_are_isolated_by_collection(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    store.add_chunks([_chunk("docA-0", "docA", "p.1", "from doc A", [1.0, 0.0, 0.0])])
    store.add_chunks([_chunk("docB-0", "docB", "p.1", "from doc B", [1.0, 0.0, 0.0])])

    results = store.query("docA", query_embedding=[1.0, 0.0, 0.0], k=5)

    assert [r["id"] for r in results] == ["docA-0"]


def test_add_chunks_with_empty_list_is_a_no_op(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    store.add_chunks([])  # must not raise


def test_get_all_chunks_returns_every_chunk_for_a_document(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    store.add_chunks(
        [
            _chunk("doc1-0", "doc1", "p.1", "penalty clause one", [1.0, 0.0, 0.0]),
            _chunk("doc1-1", "doc1", "p.2", "termination clause", [0.0, 1.0, 0.0]),
            _chunk("doc1-2", "doc1", "p.3", "penalty clause two", [0.0, 0.0, 1.0]),
        ]
    )

    all_chunks = store.get_all_chunks("doc1")

    assert {c["id"] for c in all_chunks} == {"doc1-0", "doc1-1", "doc1-2"}
    by_id = {c["id"]: c for c in all_chunks}
    assert by_id["doc1-0"]["text"] == "penalty clause one"
    assert by_id["doc1-0"]["location_label"] == "p.1"
    assert by_id["doc1-0"]["source_type"] == "pdf"


def test_get_all_chunks_on_empty_document_returns_empty_list(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))

    assert store.get_all_chunks("nonexistent-doc") == []


def test_delete_document_removes_its_chunks(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    store.add_chunks([_chunk("doc1-0", "doc1", "p.1", "payment due date", [1.0, 0.0, 0.0])])

    store.delete_document("doc1")

    assert store.get_all_chunks("doc1") == []


def test_delete_document_does_not_affect_other_documents(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    store.add_chunks([_chunk("docA-0", "docA", "p.1", "from doc A", [1.0, 0.0, 0.0])])
    store.add_chunks([_chunk("docB-0", "docB", "p.1", "from doc B", [1.0, 0.0, 0.0])])

    store.delete_document("docA")

    assert store.get_all_chunks("docA") == []
    assert [c["id"] for c in store.get_all_chunks("docB")] == ["docB-0"]


def test_delete_document_on_a_document_with_no_indexed_chunks_does_not_raise(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))

    store.delete_document("never-indexed-doc")  # must not raise

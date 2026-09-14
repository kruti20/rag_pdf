from core.chunker import chunk_segments
from core.models import ExtractedSegment


def test_empty_segments_produce_no_chunks():
    assert chunk_segments([], document_id="doc1", source_type="txt") == []


def test_short_segment_becomes_a_single_chunk():
    segments = [ExtractedSegment(text="hello world", location_label="p.1")]

    chunks = chunk_segments(segments, document_id="doc1", source_type="pdf")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == "hello world"
    assert chunk.location_label == "p.1"
    assert chunk.document_id == "doc1"
    assert chunk.source_type == "pdf"
    assert chunk.id == "doc1-0"


def test_multiple_short_segments_each_produce_their_own_chunk_with_sequential_ids():
    segments = [
        ExtractedSegment(text="first segment", location_label="p.1"),
        ExtractedSegment(text="second segment", location_label="p.2"),
    ]

    chunks = chunk_segments(segments, document_id="doc1", source_type="pdf")

    assert [c.location_label for c in chunks] == ["p.1", "p.2"]
    assert [c.text for c in chunks] == ["first segment", "second segment"]
    assert [c.id for c in chunks] == ["doc1-0", "doc1-1"]


def test_long_segment_is_split_into_multiple_overlapping_chunks():
    sentence = "The quick brown fox jumps over the lazy dog. "
    long_text = sentence * 100  # ~4600 characters, well over the chunk size
    segments = [ExtractedSegment(text=long_text, location_label="p.9")]

    chunks = chunk_segments(
        segments, document_id="doc1", source_type="pdf",
        chunk_size_chars=1000, chunk_overlap_chars=100,
    )

    assert len(chunks) > 1
    assert all(c.location_label == "p.9" for c in chunks)
    assert all(c.document_id == "doc1" for c in chunks)
    assert all(len(c.text) <= 1000 for c in chunks)
    # consecutive chunks overlap: the tail of one reappears at the head of the next
    assert chunks[0].text[-50:] in chunks[1].text

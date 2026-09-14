from core.models import Answer, Chunk, ExtractedSegment, Source


def test_chunk_holds_expected_fields():
    chunk = Chunk(
        id="doc1-0",
        document_id="doc1",
        source_type="pdf",
        location_label="p.1",
        text="hello world",
    )
    assert chunk.id == "doc1-0"
    assert chunk.document_id == "doc1"
    assert chunk.source_type == "pdf"
    assert chunk.location_label == "p.1"
    assert chunk.text == "hello world"
    assert chunk.embedding is None


def test_extracted_segment_holds_text_and_location():
    segment = ExtractedSegment(text="hello", location_label="p.1")
    assert segment.text == "hello"
    assert segment.location_label == "p.1"


def test_source_holds_location_and_snippet():
    source = Source(location_label="p.1", snippet="hello world")
    assert source.location_label == "p.1"
    assert source.snippet == "hello world"


def test_answer_holds_text_sources_and_found_flag():
    answer = Answer(
        text="The due date is the 1st.",
        sources=[Source(location_label="p.1", snippet="Payment due date")],
        found_in_document=True,
    )
    assert answer.text == "The due date is the 1st."
    assert answer.sources[0].location_label == "p.1"
    assert answer.found_in_document is True


def test_answer_defaults_to_no_sources():
    answer = Answer(text="I couldn't find that in the document.", found_in_document=False)
    assert answer.sources == []

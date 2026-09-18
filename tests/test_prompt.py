from core.prompt import build_prompt


def test_prompt_instructs_grounding_and_not_found_fallback():
    prompt = build_prompt("What is the due date?", [])

    assert "only" in prompt.lower()
    assert "i couldn't find that in the document" in prompt.lower()


def test_prompt_includes_each_chunk_with_its_location_label():
    chunks = [
        {"location_label": "p.1", "text": "Payment is due on the first."},
        {"location_label": "p.2", "text": "Late fees apply after 5 days."},
    ]

    prompt = build_prompt("What is the due date?", chunks)

    assert "p.1" in prompt
    assert "Payment is due on the first." in prompt
    assert "p.2" in prompt
    assert "Late fees apply after 5 days." in prompt


def test_prompt_includes_the_question():
    prompt = build_prompt("What is the due date?", [])
    assert "What is the due date?" in prompt


def test_prompt_with_no_chunks_has_a_placeholder_not_a_crash():
    prompt = build_prompt("Anything?", [])
    assert "no relevant context" in prompt.lower()


def test_prompt_includes_chat_history_when_provided():
    history = [("user", "What about late fees?"), ("assistant", "Late fees are $50.")]

    prompt = build_prompt("And penalties?", [], chat_history=history)

    assert "What about late fees?" in prompt
    assert "Late fees are $50." in prompt


def test_prompt_omits_history_section_when_none_provided():
    prompt = build_prompt("Anything?", [])
    assert "Conversation so far" not in prompt


def test_prompt_labels_each_chunk_with_its_document_name():
    chunks = [
        {"document_id": "doc-a", "location_label": "p.1", "text": "Payment is due on the first."},
        {"document_id": "doc-b", "location_label": "p.2", "text": "Termination requires notice."},
    ]
    document_names = {"doc-a": "ContractA.pdf", "doc-b": "ContractB.pdf"}

    prompt = build_prompt("What is the due date?", chunks, document_names)

    assert "[ContractA.pdf — p.1]" in prompt
    assert "[ContractB.pdf — p.2]" in prompt


def test_prompt_falls_back_to_bare_location_label_when_document_name_unknown():
    chunks = [{"document_id": "doc-a", "location_label": "p.1", "text": "Payment is due on the first."}]

    prompt = build_prompt("What is the due date?", chunks)

    assert "[p.1]" in prompt

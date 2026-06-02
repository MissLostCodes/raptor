from raptor.chunking.structure_parser import parse_titles_from_response


def test_plain_json_array_of_objects():
    resp = '[{"title": "Introduction"}, {"title": "Methods"}]'
    assert parse_titles_from_response(resp) == ["Introduction", "Methods"]


def test_strips_json_code_fences():
    resp = '```json\n[{"title": "Intro"}, {"title": "Results"}]\n```'
    assert parse_titles_from_response(resp) == ["Intro", "Results"]


def test_strips_bare_code_fences():
    resp = '```\n[{"title": "A"}]\n```'
    assert parse_titles_from_response(resp) == ["A"]


def test_array_of_plain_strings():
    resp = '["Intro", "Body", "Conclusion"]'
    assert parse_titles_from_response(resp) == ["Intro", "Body", "Conclusion"]


def test_extracts_array_embedded_in_prose():
    resp = 'Sure! Here is the structure:\n[{"title": "Intro"}]\nHope that helps.'
    assert parse_titles_from_response(resp) == ["Intro"]


def test_garbage_returns_empty_list():
    assert parse_titles_from_response("not json at all") == []
    assert parse_titles_from_response("") == []
    assert parse_titles_from_response("   ") == []


def test_skips_items_without_string_title():
    resp = '[{"title": "Keep"}, {"level": 1}, {"title": ""}, {"title": "Also"}]'
    assert parse_titles_from_response(resp) == ["Keep", "Also"]

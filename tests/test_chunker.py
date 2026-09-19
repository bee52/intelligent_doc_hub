from src.ingestion.chunker import chunk


def test_chunk_without_overlap() -> None:
    assert chunk("abcdefgh", 3) == ["abc", "def", "gh"]


def test_chunk_with_overlap() -> None:
    assert chunk("abcdef", 3, overlap=1) == ["abc", "cde", "ef"]


def test_chunk_smaller_than_chunk_size() -> None:
    result = chunk("short", 20)

    assert result == ["short"]
    assert result[0].metadata.index == 0
    assert result[0].metadata.start == 0
    assert result[0].metadata.end == 5


def test_chunk_by_lines() -> None:
    assert chunk("one\ntwo\nthree\n", 2, strategy="by_lines") == [
        "one\ntwo\n",
        "three\n",
    ]
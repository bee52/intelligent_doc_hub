# Chunker

The reusable chunker lives in `src/ingestion/chunker.py`.

## Command line

```powershell
python -m src.ingestion.chunker --input data/input.txt --chunk-size 500 --overlap 50
```

Use `--strategy by_lines` to chunk by newline-delimited units. Use `--output-dir chunks` to write each chunk to a separate text file instead of printing to standard output.

## Python API

```python
from src.ingestion.chunker import chunk, iter_chunks

chunks = chunk(text, chunk_size=500, overlap=50)
for item in iter_chunks(text, chunk_size=500, strategy="by_lines"):
    print(item, item.metadata.index)
```

Each result is string-compatible and exposes `metadata.index`, `metadata.start`, and `metadata.end`.

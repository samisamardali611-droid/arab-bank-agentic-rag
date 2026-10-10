import json
from pathlib import Path
from bank_rag.config import CHUNKS_DIR

def main():
    base_dir = CHUNKS_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    chunk_files = sorted(base_dir.glob("[0-9][0-9][0-9][0-9]_chunks.json"))

    if not chunk_files:
        print("❌ No *_chunks.json files found")
        return

    all_chunks = []

    for file in chunk_files:
        data = json.loads(file.read_text(encoding="utf-8"))
        all_chunks.extend(data)
        print(f"✔ Loaded {len(data)} chunks from {file.name}")

    output_file = base_dir / "all_chunks.json"
    output_file.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("-" * 40)
    print(f"✅ Total chunks: {len(all_chunks)}")
    print(f"📁 Saved to {output_file.name}")

if __name__ == "__main__":
    main()

import json
from pathlib import Path
from bank_rag.config import SECTIONS_DIR, CHUNKS_DIR

MIN_CHARS = 300
CHUNK_SIZE = 500

def chunk_text(text, chunk_size):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

def main():
    base_dir = SECTIONS_DIR
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    section_files = sorted(base_dir.glob("*_sections.json"))

    if not section_files:
        print("❌ No *_sections.json files found. Run split_all_years.py first.")
        return

    for section_file in section_files:
        year = int(section_file.stem.split("_")[0])  # 2021_sections -> 2021
        out_file = CHUNKS_DIR / f"{year}_chunks.json"

        sections = json.loads(section_file.read_text(encoding="utf-8"))
        chunks = []

        for sec in sections:
            content = (sec.get("content") or "").strip()
            if len(content) < MIN_CHARS:
                continue

            for idx, chunk in enumerate(chunk_text(content, CHUNK_SIZE)):
                chunks.append({
                    "year": year,
                    "section_title": sec.get("section_title"),
                    "note_number": sec.get("note_number"),
                    "chunk_index": idx,
                    "text": chunk
                })

        out_file.write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        print(f"✅ {year}: Created {len(chunks)} chunks → {out_file.name}")

    print("🎉 All years chunks created successfully")

if __name__ == "__main__":
    main()

import re
import json
from pathlib import Path
from bank_rag.config import CLEAN_DIR, SECTIONS_DIR

# ==============================
# Regex patterns
# ==============================
NOTE_HEADER_RE = re.compile(r"^(\d{1,2})\.\s+(.+)$")
ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z\s\-&,()]{5,}$")

def split_into_sections(text: str, year: int):
    sections = []
    current_section = {
        "year": year,
        "section_title": "Introduction",
        "note_number": None,
        "content": []
    }

    for line in text.splitlines():
        s = line.strip()

        note_match = NOTE_HEADER_RE.match(s)
        if note_match:
            if current_section["content"]:
                sections.append({
                    **current_section,
                    "content": "\n".join(current_section["content"]).strip()
                })
            current_section = {
                "year": year,
                "section_title": note_match.group(2),
                "note_number": note_match.group(1),
                "content": []
            }
            continue

        if ALL_CAPS_RE.match(s):
            if current_section["content"]:
                sections.append({
                    **current_section,
                    "content": "\n".join(current_section["content"]).strip()
                })
            current_section = {
                "year": year,
                "section_title": s.title(),
                "note_number": None,
                "content": []
            }
            continue

        current_section["content"].append(line)

    if current_section["content"]:
        sections.append({
            **current_section,
            "content": "\n".join(current_section["content"]).strip()
        })

    return sections

def main():
    base_dir = CLEAN_DIR
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    clean_files = sorted(base_dir.glob("*_clean.txt"))

    if not clean_files:
        print("❌ No *_clean.txt files found in data/clean")
        return

    for clean_file in clean_files:
        year = int(clean_file.stem.split("_")[0])  # 2021_clean -> 2021
        out_file = SECTIONS_DIR / f"{year}_sections.json"

        text = clean_file.read_text(encoding="utf-8", errors="ignore")
        sections = split_into_sections(text, year)

        out_file.write_text(
            json.dumps(sections, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        print(f"✅ {year}: Extracted {len(sections)} sections → {out_file.name}")

    print("🎉 All years sections created successfully")

if __name__ == "__main__":
    main()

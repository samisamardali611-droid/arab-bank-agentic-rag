import re
from pathlib import Path

# ==============================
# Paths (مضبوطة حسب مشروعك)
# ==============================

from bank_rag.config import OCR_DIR as RAW_DIR, CLEAN_DIR

# ==============================
# Regex patterns للتنظيف
# ==============================

PAGE_NUMBER_RE = re.compile(r"^\s*-\s*\d+\s*-\s*$")
FOOTER_RE = re.compile(
    r"The accompanying notes .* integral part .* read with them",
    re.IGNORECASE
)

# ==============================
# دالة تنظيف النص
# ==============================

def clean_text(text: str) -> str:
    lines = []

    for line in text.splitlines():
        s = line.strip()

        # حذف أرقام الصفحات مثل: - 12 -
        if PAGE_NUMBER_RE.match(s):
            continue

        # حذف الجملة المتكررة أسفل الصفحات
        if FOOTER_RE.search(s):
            continue

        # حذف أسطر قصيرة خاصة بالتدقيق (Deloitte)
        if "deloitte" in s.lower() and len(s) < 80:
            continue

        lines.append(line)

    text = "\n".join(lines)

    # تنظيف المسافات الزائدة
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # تنظيف underscores الناتجة عن OCR
    text = re.sub(r"_{2,}", " ", text)

    return text.strip() + "\n"

# ==============================
# الدالة الرئيسية
# ==============================

def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    print("Current working directory:", Path.cwd())
    print("RAW_DIR exists:", RAW_DIR.exists())
    print("RAW_DIR path:", RAW_DIR.resolve())
    print("-" * 40)

    if not RAW_DIR.exists():
        print("❌ data/ocr folder not found")
        return

    files = list(RAW_DIR.iterdir())
    print(f"Found {len(files)} files in data/ocr")

    if not files:
        print("❌ No files found inside data/ocr")
        return

    for file in files:
        if not file.name.endswith(".txt"):
            continue

        print(f"Processing: {file.name}")

        raw_text = file.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw_text)

        output_file = CLEAN_DIR / file.name.replace("_ocr", "_clean")
        output_file.write_text(cleaned, encoding="utf-8")

        print(f"✔ Cleaned: {output_file.name}")

    print("-" * 40)
    print("✅ Cleaning finished successfully")

# ==============================
# Entry point
# ==============================

if __name__ == "__main__":
    main()

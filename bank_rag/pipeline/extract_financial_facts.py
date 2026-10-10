import re
import json
from pathlib import Path

from bank_rag.config import CLEAN_DIR as DATA_DIR, DATA_DIR as OUTPUT_DIR
YEARS = ["2021", "2022", "2023", "2024"]

def normalize_number(s: str) -> str:
    # remove spaces and OCR junk
    return re.sub(r"[^\d\.\-]", "", s)

def extract_value(patterns, text):
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return normalize_number(m.group(1))
    return None

def extract_year_facts(year):
    file = DATA_DIR / f"{year}_clean.txt"
    if not file.exists():
        return None

    text = file.read_text(encoding="utf-8", errors="ignore")

    facts = {
        "year": year,
        "total_assets": extract_value([
            r"TOTAL\s+ASSETS\s+([\d\s\.\-]+)",
            r"Total Assets\s+([\d\s\.\-]+)"
        ], text),
        "total_liabilities": extract_value([
            r"TOTAL\s+LIABILITIES\s+([\d\s\.\-]+)",
            r"Total Liabilities\s+([\d\s\.\-]+)"
        ], text),
        "net_profit": extract_value([
            r"Net Profit\s*\(Loss\)\s+([\d\s\.\-\(\)]+)",
            r"Net Profit\s+([\d\s\.\-]+)"
        ], text),
        "total_income": extract_value([
            r"Total Income\s+([\d\s\.\-]+)"
        ], text)
    }

    return facts

def main():
    all_facts = {}

    for y in YEARS:
        facts = extract_year_facts(y)
        if facts:
            all_facts[y] = facts
            print(f"✔ Extracted facts for {y}")
        else:
            print(f"✖ No data for {y}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "financial_facts.json"
    out.write_text(json.dumps(all_facts, indent=2), encoding="utf-8")

    print("\nSaved to financial_facts.json")

if __name__ == "__main__":
    main()

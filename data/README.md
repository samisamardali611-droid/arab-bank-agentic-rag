# Corpus provenance

The supplied OCR text is derived from Arab Bank Group financial statements for 2021–2024. The corresponding reports are listed on the bank's official [Annual Reports page](https://www.arabbank.com/mainmenu/home/investor-relations/annual-reports).

| Source report | OCR input | Entries in supplied index metadata |
| --- | --- | ---: |
| Arab Bank Group Financial Statements 2021 | `ocr/2021_ocr.txt` | 121 |
| Arab Bank Group Financial Statements 2022 | `ocr/2022_ocr.txt` | 139 |
| Arab Bank Group Financial Statements 2023 | `ocr/2023_ocr.txt` | 143 |
| Arab Bank Group Financial Statements 2024 | `ocr/2024_ocr.txt` | 129 |
| **Total** | **4 files** | **532** |

OCR inputs and the precomputed FAISS index were retained from the submitted project. The PDF-to-OCR method and settings are not included in that submission. The index metadata contains the text for each vector and preserves year, section title, and note number when available; it has no reliable page coordinates.

The PDF copies, duplicate merged corpus, and generated intermediate files were omitted from this prepared repository. Original PDFs can be obtained from the source page; cleanup, sectioning, chunking, and index-generation code are included.

These are third-party report excerpts, not synthetic or customer-account data. Their original rights remain with their respective owners; this repository adds no redistribution license. Review the reports' terms before public redistribution. If necessary, keep the dataset and index private and publish only the code with source instructions.

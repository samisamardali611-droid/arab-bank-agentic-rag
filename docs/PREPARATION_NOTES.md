# Repository preparation notes

This copy was prepared from the supplied RAG submission. The original archive was left unchanged.

## Changes

- Replaced the old README, which contained a credential-shaped OpenAI key, with credential-free setup instructions and a description grounded in the code.
- Added `.gitignore` and an empty `.env.example`; loaded local configuration centrally.
- Organized Python code into the `bank_rag` package, corrected internal imports, and replaced working-directory-dependent paths with project-root paths.
- Added a root Streamlit entry point with a missing-key setup state and a missing-index message.
- Deferred API client creation so local imports and offline data preparation do not require a key.
- Set the existing `gpt-4.1-mini` model explicitly on SDK agents rather than depending on SDK defaults.
- Corrected chunk merging so `all_chunks.json` is not accidentally merged into itself on later runs.
- Added guards for malformed planner output, absent FAISS neighbors, and unsupported numeric regex patterns.
- Preserved the supplied OCR inputs, index, and aligned metadata. Excluded raw PDF copies, duplicate corpus files, and generated intermediate outputs.
- Omitted the old submission PDF because parts of it describe a previous folder layout and stronger behavior than the code implements. The README documents current behavior and limitations.
- Pinned direct dependency versions and added targeted offline checks.

## Scope of verification

Preparation checks use local dependencies and artifacts. No live OpenAI query or embedding rebuild is required by the tests. Numeric extraction, source attribution, and multi-year comparison remain prototype features with the limitations described in the README.

Observed results on 10 October 2026: all 8 offline tests passed with Python 3.12 on Linux; all 22 package submodules imported without an API key; the four OCR inputs regenerated exactly the same 532 chunk records; a repeated merge was byte-identical. FAISS loaded the supplied 532-vector, 3,072-dimension index, with aligned metadata. A text-pattern scan and an exact check for the credential found in the original README found no copy of it in the prepared deliverable.

The presence of an exposed credential in the source archive does not prove whether it remains active. Its owner should revoke it if active; the prepared repository neither uses nor includes it.

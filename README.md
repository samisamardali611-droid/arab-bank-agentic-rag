# Arab Bank Financial Statements - Agentic RAG

A Python prototype for exploring Arab Bank Group financial statements from **2021–2024** through a Streamlit chat interface. It combines OpenAI embeddings, FAISS search, keyword reranking, and an agent workflow for question planning and evidence retrieval.

The project demonstrates a document-to-search pipeline and an experimental approach to numeric extraction. It is a portfolio project, not an official bank service or a production financial analysis system.

## What is implemented

- OCR-text cleanup, section detection, and fixed-size word chunking.
- A supplied FAISS `IndexFlatL2` index containing **532 vectors of 3,072 dimensions**, with matching text metadata.
- Semantic retrieval followed by a weighted vector/keyword reranker.
- OpenAI Agents SDK planning and retrieval agents with function tools.
- Pure Python numeric extraction and a fallback for table-like text.
- A Streamlit interface that displays the current conversation.
- Separate baseline scripts for inspecting retrieval and an earlier Q&A approach.

## Technology and parameters

| Component | Implementation |
| --- | --- |
| Language | Python 3.11+; preparation checks used Python 3.12 on Linux |
| Interface | Streamlit |
| Agent orchestration | OpenAI Agents SDK |
| Chat model | `gpt-4.1-mini` |
| Embedding model | `text-embedding-3-large` |
| Vector search | FAISS `IndexFlatL2` |
| Chunking | Up to 500 words; no overlap; sections below 300 characters are skipped |
| Retrieval | Up to 80 candidates, reranked to 12 |
| Context trimming | Each retrieved chunk is truncated to 1,200 characters |

The reranker combines `0.7 * vector_score + 0.3 * keyword_score`. Keyword scores include phrase boosts; they are not normalized probabilities. Character limits reduce context length but do not guarantee a token budget or prevent rate-limit errors.

## Quick start

Open a terminal in the project folder after cloning or extracting it. A working OpenAI API account and a key with access to the configured models are required for live questions. API usage may incur charges.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` locally and set `OPENAI_API_KEY` to your own new key. Keep `.env` out of GitHub. Then run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` as described above, then run:

```bash
.venv/bin/python -m streamlit run app.py
```

Use Python 3.11 or newer. Open the local URL printed by Streamlit. The supplied index allows you to skip rebuilding the corpus. If no key is configured, the app displays a setup message without making an API request.

`requirements.txt` pins the direct dependencies used for preparation checks. It is not a complete transitive dependency lockfile. Windows execution and live API answers have not been verified as part of this preparation.

## Example questions

- What accounting policies are described in the 2024 statements?
- How is credit risk discussed in the financial statements?
- What is the total assets figure in the 2024 statement?
- What is the earnings per share figure in 2022?

These are prompts for exploration, not evaluated examples with guaranteed answers. Check numeric responses against the original report, including units, year, and table column.

## How the workflow operates

1. The UI sends the current question to the orchestrator.
2. The triage agent calls a planner tool to classify the question and identify years and a metric.
3. The retrieval agent can call a tool that embeds the query, searches FAISS, reranks the results, and returns bounded text excerpts.
4. For numeric questions, the orchestrator calls regex-based verification directly, then tries table-text extraction if necessary.
5. For descriptive questions, the model answers using up to 2,000 characters of the retrieval agent's output.

The verifier and table extractor have SDK tool wrappers, but the main orchestrator invokes their Python functions directly. The UI keeps visible history; previous messages are not passed to the backend as conversational context.

## Repository contents

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit entry point |
| `bank_rag/config.py` | Root-relative paths and shared model names |
| `bank_rag/client.py` | Lazy API client creation |
| `bank_rag/agents/` | Planner, retrieval, verification tools, and orchestrator |
| `bank_rag/engine/` | Baseline RAG engine and keyword reranker |
| `bank_rag/pipeline/` | Data preparation and index-building scripts |
| `bank_rag/baselines/` | Earlier CLI experiments |
| `data/ocr/` | Four supplied OCR-text inputs |
| `data/README.md` | Source provenance and corpus details |
| `artifacts/` | Supplied index and aligned text metadata |
| `tests/` | Offline regression checks |
| `docs/PREPARATION_NOTES.md` | Changes made while preparing this repository |

## Rebuild from the supplied OCR text

The original PDF-to-OCR step is not included. The reproducible pipeline starts with the supplied `data/ocr/YYYY_ocr.txt` files. See [data sources](data/README.md) for the original reports.

Run the following from the repository root using your virtual environment's Python. On Windows, replace `python` with `.\.venv\Scripts\python.exe` if the environment is not activated; on macOS/Linux, use `.venv/bin/python`.

```bash
python -m bank_rag.pipeline.clean_ocr
python -m bank_rag.pipeline.split_all_years
python -m bank_rag.pipeline.chunk_all_years
python -m bank_rag.pipeline.merge_chunks
```

These stages run locally without an API key. The generated files go under `data/clean`, `data/sections`, and `data/chunks`, which are excluded from Git.

The next command **calls the embeddings API and replaces the local index and metadata**. Back up the existing `artifacts` folder if you want to retain the supplied index.

```bash
python -m bank_rag.pipeline.embed_chunks
```

Use the same embedding model and dimensions for corpus and query vectors. Restart the app after rebuilding because the baseline engine caches its index. An optional regex-only extraction experiment can be run with:

```bash
python -m bank_rag.pipeline.extract_financial_facts
```

## Validation and current limitations

Run the offline checks after installing requirements:

```bash
python -m unittest discover -s tests -v
```

The checks cover module imports without a key, FAISS/index metadata consistency, path independence, repeatable chunk merging, defensive parsing, missing-neighbor handling, and the initial UI state. They do **not** measure financial accuracy, retrieval quality, model response quality, or live API compatibility.

During preparation on 10 October 2026, all **8 offline tests passed**. Running the local OCR-cleaning-to-chunking pipeline reproduced the submitted 532 chunks exactly; repeating the merge did not duplicate them. The supplied embeddings were retained rather than regenerated.

Known limitations retained from the original prototype:

- Numeric extraction uses heuristics. It can confuse nearby numbers, signs, units, and table columns. Its `verified` statuses and confidence labels are program labels, not statistical or financial validation.
- The main retrieval tool does not enforce the planner's year constraint. Numeric regex matching is not year-aware.
- Comparison questions enter a numeric branch that can return a single match; reliable multi-year comparison is not yet implemented in the main agent workflow.
- Retrieval-agent output is model-generated and may change the presentation of tool evidence. The main orchestrator passes an empty structured source list to the formatter; some evidence snippets contain year/section labels, but complete page-level citations are not implemented.
- Flattening chunks and truncating context can remove table structure or relevant evidence.
- No labeled evaluation set, measured accuracy, latency benchmark, or production deployment is included.

Potential next steps are year-filtered retrieval, structured evidence passed directly to the verifier, robust table parsing, and a labeled evaluation set before making accuracy claims.

## Data, credentials, and attribution

The source reports are public financial statements listed by [Arab Bank](https://www.arabbank.com/mainmenu/home/investor-relations/annual-reports). Public availability is not a license grant; third-party report content retains its original rights. This repository does not claim bank endorsement. No project-wide license is selected in this prepared copy.

Keep credentials in environment variables or a local `.env`. The `.env.example` file contains no credential. The optional tracing setting disables Agents SDK tracing for local trials. If a key was previously shared, rotate it before use; deleting its text from a file does not revoke it.

References: [OpenAI agent definitions](https://developers.openai.com/api/docs/guides/agents/define-agents), [model configuration](https://developers.openai.com/api/docs/guides/agents/models), and [API credential guidance](https://developers.openai.com/api/docs/guides/production-best-practices).

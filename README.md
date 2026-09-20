# Domain-Specific RAG Chatbot for PDF Question Answering

A chatbot that answers questions **only from PDF documents you upload** (company policies, course notes, manuals, legal or training material). It uses Retrieval-Augmented Generation (RAG): it finds the most relevant passages in your documents, gives them to a language model, and shows the **source document and page number** for every answer. If the answer is not in the documents, it says so instead of guessing.

> Student project - Unlox Academy, *AI Major Project 3: Domain-Specific RAG Chatbot*.
> **Author:** _your name_ &nbsp;|&nbsp; **GitHub:** _your repository link_ &nbsp;|&nbsp; **Live app:** _your deployed link_

## Features

- Upload one or more PDFs; validates file type, real PDF content and size (default max 10 MB per file)
- Page-by-page text extraction with `pypdf`; empty pages are skipped safely
- Chunking (800 characters, 120 overlap) with the file name and page number kept as metadata
- Embeddings with Sentence Transformers `all-MiniLM-L6-v2`, stored in a FAISS vector database
- Top-4 retrieval with a relevance threshold, so unrelated questions are refused without calling the LLM
- Strict "answer only from the context" prompt with a clear fallback message
- Protection against prompt injection: instructions hidden inside documents are ignored
- Sources (document, page, relevance, snippet) shown under every answer
- Streamlit interface: sidebar uploader, **Process Documents**, chat history, **Clear Chat**, **Clear Documents**
- Optional: save the FAISS index locally and reload it later
- Built-in evaluation script and a 26-question test sheet

## Architecture

![Architecture diagram](docs/architecture.png)

```
Upload PDF -> extract text per page -> split into chunks -> embed chunks -> FAISS index
User question -> embed question -> retrieve top chunks -> guardrail prompt + context -> Gemini -> answer + sources
```

## Project structure

```
domain_rag_chatbot/
|-- app.py                # Streamlit interface (Module 7)
|-- rag_pipeline.py       # retrieval + answer generation (Modules 5 and 6)
|-- document_loader.py    # upload validation + text extraction (Modules 1 and 2)
|-- vector_store.py       # chunking, embeddings, FAISS, save/load (Modules 3 and 4)
|-- prompt.py             # guardrail prompt and fallback message
|-- requirements.txt
|-- README.md
|-- .env.example          # copy to .env and add your API key
|-- .gitignore
|-- .streamlit/config.toml
|-- documents/            # sample PDFs (fictional company) for testing
|   |-- Company_Policy.pdf
|   |-- Employee_Handbook.pdf
|   `-- Facilities_Notice.pdf
|-- vector_store/         # saved_index/ is created here if you choose to save the index
|-- docs/architecture.png
`-- tests/
    |-- test_questions.csv   # 26-question testing sheet
    `-- evaluate.py          # fills in retrieved sources, refusal checks and timings
```

## Setup

**Requirements:** Python 3.10 or newer, an internet connection for the first run (the embedding model, about 90 MB, is downloaded once), and a free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
# 1. Get the code
git clone <your-repository-link>
cd domain_rag_chatbot

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env               # Windows: copy .env.example .env
# open .env and set GOOGLE_API_KEY=your_real_key

# 5. Run
streamlit run app.py
```

Never commit `.env` to GitHub. It is already listed in `.gitignore`.

## How to use

1. In the sidebar, upload one or more PDFs (try the files in `documents/`).
2. Click **Process Documents** and wait for the "Indexed ... PDF file(s)" message.
3. Ask a question in the chat box, for example *"How is attendance calculated?"*
4. Open **Sources** under the answer to see the document, page and matching text.
5. **Clear Chat** removes the conversation. **Clear Documents** removes the indexed documents so you can upload a different set.

Tick **Save index on this computer for reuse** before processing if you want a **Load saved index** button next time. Leave it off on shared or deployed servers, because the saved index contains text from your documents.

## Configuration (optional, in `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `GOOGLE_API_KEY` | - | Required. Your Gemini API key |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Gemini model name |
| `TOP_K` | `4` | Number of chunks retrieved per question (guidance suggests 3-5) |
| `MIN_RELEVANCE` | `0.20` | Minimum cosine similarity (0-1) for a chunk to count as relevant |
| `MAX_FILE_MB` | `10` | Maximum size of one PDF. Keep `maxUploadSize` in `.streamlit/config.toml` in sync |

## Testing and evaluation

`tests/test_questions.csv` has 26 questions about the sample documents: 16 direct questions, 4 paraphrased questions, 1 prompt-injection test and 5 questions whose answer is **not** in the documents.

```bash
python tests/evaluate.py          # retrieval only: no API key needed
python tests/evaluate.py --llm    # also generates answers with Gemini
```

The script indexes `documents/`, runs every question and writes `tests/results.csv` with:

- **Retrieved Source** and **Retrieval OK?** - did the correct document and page appear in the top results, and at which rank
- **Answer**, **Refused?** and **Auto Check** (with `--llm`) - checks that unavailable questions are refused and that the injected "HACKED" instruction is ignored
- **Seconds** - response time per question

Answer correctness still needs a human read: compare each answer with the **Notes** column and fill in **Correct?**. If real questions are wrongly refused, lower `MIN_RELEVANCE`; if unrelated questions still reach the LLM, raise it.

## Deployment (Streamlit Community Cloud)

1. Push the project to GitHub (without `.env`).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app from your repository with `app.py` as the main file.
3. Open **Advanced settings -> Secrets** and add: `GOOGLE_API_KEY = "your_real_key"`
4. Deploy, then put the app link at the top of this README.

## Responsible AI and security

- The API key is read from `.env` or Streamlit secrets and is never in the code or the repository.
- Only PDF files up to the size limit are accepted; content is checked, not just the file extension.
- The prompt tells the model to treat document text as untrusted data and to ignore instructions inside it.
- The app shows a reminder to verify high-stakes information. **Answers can still be wrong**: retrieval may pick the wrong passage, and the model may misread it. Always check important answers against the source page.
- Do not upload confidential documents without permission.

## Limitations

- Scanned (image-only) PDFs are not supported because OCR is not included.
- Tables and multi-column layouts may be extracted in a confusing order.
- Follow-up questions are treated independently (there is no conversation memory).
- The sample PDFs describe a fictional company.

## Troubleshooting

| Problem | What to do |
|---|---|
| "GOOGLE_API_KEY is not set" | Create `.env` from `.env.example`, add your key, restart Streamlit |
| Model not found / not available error | Set `GEMINI_MODEL` in `.env` to a model shown in Google AI Studio |
| Quota or rate limit (429) error | Wait a minute; with `evaluate.py --llm` increase `--delay` |
| First "Process Documents" is slow | The embedding model is downloading; later runs are faster |
| "no extractable text" for a file | It is probably a scanned PDF; use a text-based PDF |

## Tech stack

Python, pypdf, LangChain (text splitters, community FAISS, HuggingFace and Google GenAI integrations), Sentence Transformers (`all-MiniLM-L6-v2`), FAISS, Google Gemini, Streamlit, python-dotenv.

# Document Q&A — MCP Server
**Nexla Software Engineer Take-Home Assignment**
*Siyan Shaikh*

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js (only needed for MCP Inspector testing)

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd nexla-assignment
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install pymupdf scikit-learn numpy fastmcp "mcp[cli]"
```

### 4. Add the provided PDF files
```
data/
├── 20220405.pdf
├── 20220606.pdf
├── 20221013.pdf
├── 20230115.pdf
└── 20230317.pdf
```

### 5. Start the MCP server
```bash
python server.py
```

Expected output:
```
Loading PDFs...
[ingest] Loaded 5 PDFs → 35 chunks
Creating embeddings...
[embed] Vocabulary size : 2,638
[embed] Matrix shape    : (35, 2638)
Server ready!
```

The server runs over SSE at `http://127.0.0.1:8000`.

### 6. (Optional) Test with MCP Inspector
In a second terminal:
```bash
npx @modelcontextprotocol/inspector
```
Set Transport to `SSE`, URL to `http://127.0.0.1:8000/sse`, click Connect.

### 7. (Optional) Run the CLI instead
```bash
python main.py
```

---

## Architecture Overview

```
data/ (5 PDFs)
     │
     ▼
src/ingest.py    →   PyMuPDF page extraction
                     Artifact cleaning (watermark token removal)
                     400-word overlapping chunks (80-word overlap)
     │
     ▼
src/embed.py     →   TF-IDF vectorizer
                     sublinear_tf, bigrams, news stopwords
                     Sparse matrix (chunks × vocabulary)
     │
     ▼
src/query.py     →   Hybrid scoring (cosine sim + keyword coverage)
                     Source capping (max 2 chunks per document)
                     Near-duplicate removal (Jaccard similarity)
                     MMR reranking for diversity
     │
     ▼
server.py        →   FastMCP server over SSE
                     Exposes query_documents_tool
                     MCP-compliant tool schema
```

### Key design decisions and why

**No vector database.** With 5 PDFs and ~35 chunks, a full vector DB (Chroma, FAISS) adds infrastructure overhead with no retrieval benefit at this scale. Scikit-learn's sparse TF-IDF matrix fits entirely in memory and is faster to iterate on.

**No LLM for answer generation.** The assignment asks for grounded answers derived from document content. Returning the actual chunk text with source attribution is more auditable and avoids hallucination risk without adding API cost or latency.

**Overlapping chunks instead of whole pages.** A newspaper front page covers 10+ stories. Treating a whole page as one chunk means every query matches it regardless of topic. 400-word chunks with 80-word overlap give the retriever topic-level granularity while preserving cross-boundary context.

**MMR reranking.** Pure cosine similarity returns the most similar chunks, which are often near-duplicates. Maximal Marginal Relevance balances relevance against redundancy, ensuring results span distinct parts of the corpus.

---

## Tool Documentation

### `query_documents_tool`

Accepts a natural language question and returns grounded answers with source attribution from the indexed PDFs.

**Input**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `question` | string | yes | — | Natural language question |
| `top_k` | integer | no | 3 | Number of chunks to retrieve |

**Output**

```json
{
  "question": "string",
  "results": [
    {
      "answer": "string — relevant chunk text from the document",
      "source": "string — PDF filename",
      "page":   "integer — 1-based page number",
      "score":  "float — hybrid relevance score (0–1)"
    }
  ]
}
```

**Example queries**
- `"What is happening in Ukraine?"`
- `"What is the economic state?"`
- `"What is the situation with climate change?"`

---

## Example Interaction Log

All three interactions were run via MCP Inspector connected over SSE (`http://127.0.0.1:8000/sse`).

---

### Query 1 — "What is happening in Ukraine?"

**Input**
```json
{ "question": "What is happening in Ukraine?", "top_k": 3 }
```

**Output**
```json
{
  "question": "What is happening in Ukraine?",
  "results": [
    {
      "answer": "Ukraine. IVOR PRICKETT FOR THE NEW YORK TIMES Continued on Page A9 Tatyana Petrovna found three bodi...",
      "source": "20220405.pdf",
      "page": 1,
      "score": 0.2377
    },
    {
      "answer": "was done for immigration purposes. It remains unclear who else, if anyone, learned about the back gr...",
      "source": "20230115.pdf",
      "page": 1,
      "score": 0.2257
    }
  ]
}
```

---

### Query 2 — "What is the economic state?"

**Input**
```json
{ "question": "What is the economic state?", "top_k": 3 }
```

**Output**
```json
{
  "question": "What is the economic state?",
  "results": [
    {
      "answer": "In an extraordinary effort to stave off financial contagion and reassure the world that the American banking system is sound...",
      "source": "20230317.pdf",
      "page": 1,
      "score": 0.3012
    },
    {
      "answer": "Federal Reserve officials raised interest rates again as they continue to battle the highest inflation in four decades...",
      "source": "20220606.pdf",
      "page": 1,
      "score": 0.2741
    }
  ]
}
```

---

### Query 3 — "What is the situation with climate change?"

**Input**
```json
{ "question": "What is the situation with climate change?", "top_k": 3 }
```

**Output**
```json
{
  "question": "What is the situation with climate change?",
  "results": [
    {
      "answer": "Nations need to move away much faster from fossil fuels to retain any hope of preventing a perilous future on an overheated planet, according to a major new report on climate change...",
      "source": "20221013.pdf",
      "page": 1,
      "score": 0.4103
    }
  ]
}
```

---

## Vibe Coding Section

### Tools used
- **ChatGPT (GPT-4o)** — primary assistant throughout the project
- **Claude (Anthropic)** — used in parallel, especially for retrieval logic and debugging

### How I prompted and directed the AI

I described symptoms rather than asking for code. When retrieval was returning the same 3 PDFs for every query regardless of the question, I pasted the actual terminal output and asked what was wrong. This produced a specific diagnosis — a watermark artifact token (`UD54G1Dy`) present in nearly every chunk was making all documents look identical to the vectorizer — rather than a generic code suggestion.

For architecture decisions I asked "why" before accepting suggestions. When MMR reranking was proposed I asked for an explanation of the lambda parameter before applying it. When `min_df=3` caused the vocabulary to collapse to 600 terms on a 35-chunk corpus, I caught it from the debug output and pushed back until a correct value was found.

### Where I leaned on AI vs. where I overrode it

**Leaned on AI for:** vectorizer parameter tuning for a news-specific corpus, explaining MCP stdio vs SSE transport differences, diagnosing why the inspector was timing out during the handshake, and structuring the three-stage retrieval pipeline (hybrid scoring → dedup → MMR).

**Overrode AI when:** it suggested OpenAI embeddings — I kept the system fully local. When `min_df=3` killed most of the vocabulary at this corpus size I changed it to `min_df=2`. When the generated `server.py` used a `FastMCP` constructor argument (`description=`) that had been removed in the installed version, I caught it at runtime and fixed it.

### Overall view on AI in a software engineering workflow

AI tooling is most valuable as a compression on the diagnosis loop, not as a code generator. The biggest time savings came from pasting broken output and getting a structured explanation of the failure mode — faster than reading documentation cold. The risk is treating generated code as correct without understanding it. Every suggestion I applied, I read before running. The failures happened precisely where I skipped that step.

For a role that requires owning outcomes with customers under ambiguity, that discipline matters more than throughput. AI raises the floor on how quickly you can get something working; it does not raise the ceiling on whether what you built is correct. The right mental model is a fast, knowledgeable junior who needs review — not an authority who can be trusted unconditionally.

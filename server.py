from fastmcp import FastMCP

from src.ingest import load_pdfs
from src.embed import create_embeddings
from src.query import query_documents
 
# ---------------------------------------------------------------------------
# Boot – load and index all PDFs once at startup
# ---------------------------------------------------------------------------
 
print("Loading PDFs...")
docs = load_pdfs("data")
 
print("Creating embeddings...")
embeddings, vectorizer = create_embeddings(docs)
 
print("Server ready!")
 
# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
 
mcp = FastMCP("document-qa")
 
 
@mcp.tool()
def query_documents_tool(question: str, top_k: int = 3) -> dict:
    """
    Answer a natural language question using content from the indexed PDFs.
 
    Parameters
    ----------
    question : The question to answer.
    top_k    : Number of source chunks to retrieve (default 3).
 
    Returns
    -------
    A dict with:
      - question  : the original question
      - results   : list of { answer, source, page, score }
    """
    results = query_documents(question, docs, embeddings, vectorizer, top_k=top_k)
 
    if not results:
        return {
            "question": question,
            "results": [],
            "message": "No relevant content found in the indexed documents.",
        }
 
    return {
        "question": question,
        "results": [
            {
                "answer": r["text"],
                "source": r["source"],
                "page": r["page"],
                "score": r.get("score", None),
            }
            for r in results
        ],
    }
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8000) # runs over stdio by default — correct for MCP clients
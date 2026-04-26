from src.ingest import load_pdfs
from src.embed import create_embeddings
from src.query import query_documents

print("Loading PDFs...")
docs = load_pdfs("data")

print("Creating embeddings...")
embeddings, vectorizer = create_embeddings(docs)

print("Ask a question (type 'exit' to quit):")

while True:
    question = input(">> ")

    if question.lower() == "exit":
        break

    results = query_documents(question, docs, embeddings, vectorizer)

    print("\nAnswer:\n")

    if len(results) == 0:
        print("No relevant results found.\n")
        continue

    for i, r in enumerate(results):
        print(f"{i+1}. {r['text'][:200]}...\n")

    print("Sources:")
    for r in results:
        print(f"- {r['source']} (Page {r['page']})")

    print("\n" + "="*60 + "\n")
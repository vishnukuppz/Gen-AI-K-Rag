from pathlib import Path
from rag_components.json_doc_parser import load_json_documents
from rag_components.text_splitter import split_documents
from rag_components.entity_extraction import get_entity_graphs
from rag_components.neo_4j_graph import add_entity_graphs

# Resolve path to tn_government_schemes.json located in the project root
json_path = Path(__file__).resolve().parent / "tn_government_schemes.json"

print("Converting the json to document")
docs = load_json_documents(str(json_path))
print(f"Loaded {len(docs)} documents.")

print("Splitting the documents")
split_docs = split_documents(docs)
print(f"Created {len(split_docs)} split chunks.")


print(f"\nGetting the entity graphs for {len(split_docs)} chunk(s)...")
graph_docs = get_entity_graphs(split_docs)


print(f"\nAdding entity graphs to Neo4j...")
add_entity_graphs(graph_docs)

print("Entity graphs added to Neo4j successfully.")

import sys
from typing import Dict
from rag_components.graph_retriever import (
    extract_entities_and_relationships,
    match_neo4j_candidates,
    generate_cypher_query,
    execute_cypher_and_synthesize,
)


def krag_retrieval_pipeline(question: str, verbose: bool = True) -> Dict:
    """
    Main Knowledge Graph RAG retrieval pipeline:
    - Step 1: Convert user question into entities and relationships.
    - Step 2: Match extracted entities and relationships with those in Neo4j.
    - Step 3: Pass question + matched graph info to LLM to generate Cypher query.
    - Step 4: Execute query in Neo4j and pass graph data + question to LLM for final answer.
    """
    # Step 1: Extract entities and relationships from question
    if verbose:
        print("=" * 80)
        print(f"[Step 1] Extracting entities and relationships from question:\n'{question}'")
    extracted = extract_entities_and_relationships(question)
    if verbose:
        print(f"-> Entities: {extracted['entities']}")
        print(f"-> Search Keywords: {extracted['search_keywords']}")
        print(f"-> Inferred Relationships: {extracted['relationships']}")

    # Step 2: Match extracted entities and relationships with Neo4j
    if verbose:
        print("\n" + "=" * 80)
        print("[Step 2] Matching entities and relationships in Neo4j...")
    matched = match_neo4j_candidates(extracted)
    if verbose:
        print(f"-> Matched Nodes Count: {len(matched['matched_nodes'])}")
        for n in matched["matched_nodes"][:6]:
            print(f"   • {n['id']} (Labels: {n.get('labels', [])})")
        print(f"-> Connected Relationships Sample ({len(matched['connected_relationships'])} found):")
        for r in matched["connected_relationships"][:4]:
            print(f"   • ({r['source']})-[:{r['relationship']}]->({r['target']})")

    # Step 3: Generate Cypher query with LLM
    if verbose:
        print("\n" + "=" * 80)
        print("[Step 3] Passing question, entities, and Neo4j schema to LLM to generate Cypher query...")
    cypher_query = generate_cypher_query(question, extracted, matched)
    if verbose:
        print("-> Generated Cypher Query:")
        print(cypher_query)

    # Step 4: Execute Cypher in Neo4j and synthesize final output with LLM
    if verbose:
        print("\n" + "=" * 80)
        print("[Step 4] Executing Cypher query against Neo4j and generating final output with LLM...")
    execution = execute_cypher_and_synthesize(
        question,
        cypher_query,
        matched_nodes=matched.get("matched_nodes"),
    )
    if verbose:
        print(f"-> Retrieved {execution['results_count']} records from Neo4j.")
        print("\n" + "=" * 80)
        print("FINAL ANSWER:\n")
        print(execution["final_answer"])
        print("=" * 80)

    return {
        "question": question,
        "step_1_extracted": extracted,
        "step_2_matched": matched,
        "step_3_cypher": cypher_query,
        "step_4_execution": execution,
    }


if __name__ == "__main__":
    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Explain schema which has helps watering the crops?"
    )
    krag_retrieval_pipeline(query, verbose=True)
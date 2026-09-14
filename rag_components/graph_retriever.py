import os
import re
from typing import Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from rag_components.neo_4j_graph import _get_neo4j_graph

load_dotenv()


class QuestionAnalysis(BaseModel):
    """Structured extraction of entities, search keywords, and relationships from a user query."""
    entities: List[str] = Field(
        description="Key entities, domain terms, or scheme subjects mentioned in the question"
    )
    search_keywords: List[str] = Field(
        description="Search keywords, root terms, or synonyms to find matching nodes in graph (e.g., water, crop, irrigation, subsidy)"
    )
    relationships: List[str] = Field(
        description="Inferred graph relationship types (e.g., BENEFITS, PROVIDES, SPONSORED_BY, MANAGED_BY, INCLUDES, PURPOSE_FOR_LOAN)"
    )


def _get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model_name=os.getenv("OPENAI_GENERATIVE_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=temperature,
    )


def extract_entities_and_relationships(question: str, llm: Optional[ChatOpenAI] = None) -> Dict:
    """
    Step 1: Extract entities, search keywords, and relationships from the user question.
    """
    if llm is None:
        llm = _get_llm(temperature=0.0)

    structured_llm = llm.with_structured_output(QuestionAnalysis)
    analysis: QuestionAnalysis = structured_llm.invoke(question)

    return {
        "question": question,
        "entities": analysis.entities,
        "search_keywords": analysis.search_keywords,
        "relationships": analysis.relationships,
    }


def match_neo4j_candidates(
    extracted: Dict,
    graph: Optional[Neo4jGraph] = None,
    limit_nodes: int = 15,
) -> Dict:
    """
    Step 2: Match extracted entities and keywords against nodes and relationships available in Neo4j.
    """
    if graph is None:
        graph = _get_neo4j_graph()

    # Combine entities and search keywords for candidate node lookup
    all_terms = list(dict.fromkeys(extracted.get("entities", []) + extracted.get("search_keywords", [])))

    matched_nodes = []
    if all_terms:
        # Search for nodes whose id contains any of the search terms
        node_query = """
        UNWIND $terms AS term
        MATCH (n:__Entity__)
        WHERE toLower(n.id) CONTAINS toLower(term)
        RETURN DISTINCT n.id AS id, labels(n) AS labels
        LIMIT $limit
        """
        matched_nodes = graph.query(node_query, {"terms": all_terms, "limit": limit_nodes})

    # Retrieve relationships connected to the matched nodes
    node_ids = [n["id"] for n in matched_nodes if n.get("id")]
    connected_relationships = []
    if node_ids:
        rel_query = """
        UNWIND $node_ids AS nid
        MATCH (s:__Entity__ {id: nid})-[r]->(target)
        RETURN DISTINCT s.id AS source, type(r) AS relationship, labels(target) AS target_labels, target.id AS target
        LIMIT 25
        """
        connected_relationships = graph.query(rel_query, {"node_ids": node_ids[:10]})

    return {
        "terms_searched": all_terms,
        "matched_nodes": matched_nodes,
        "connected_relationships": connected_relationships,
    }


def generate_cypher_query(
    question: str,
    extracted: Dict,
    matched: Dict,
    graph: Optional[Neo4jGraph] = None,
    llm: Optional[ChatOpenAI] = None,
) -> str:
    """
    Step 3: Generate an executable, read-only Cypher query using the question, entities, and matched Neo4j context.
    """
    if llm is None:
        llm = _get_llm(temperature=0.0)

    matched_node_list = [f"- {n['id']} ({', '.join(n.get('labels', []))})" for n in matched.get("matched_nodes", [])]
    matched_nodes_str = "\n".join(matched_node_list) if matched_node_list else "None found."

    relationships_sample = [
        f"({r.get('source')})-[:{r.get('relationship')}]->({r.get('target')})"
        for r in matched.get("connected_relationships", [])[:10]
    ]
    relationships_sample_str = "\n".join(relationships_sample) if relationships_sample else "None found."

    prompt = f"""You are a Neo4j Cypher query expert.
Write an executable Cypher READ query (MATCH ... RETURN) to extract relevant knowledge graph data to answer the user question.

User Question:
"{question}"

Extracted Entities & Keywords:
{extracted.get('entities', []) + extracted.get('search_keywords', [])}

Extracted Potential Relationships:
{extracted.get('relationships', [])}

Relevant Matched Nodes in Neo4j:
{matched_nodes_str}

Sample Graph Patterns in Neo4j:
{relationships_sample_str}

Neo4j Schema Guidelines:
1. Primary entity nodes have label `__Entity__` and specific labels like `Scheme`, `Department`, `Beneficiary`, `Purpose`, `Item`, etc.
2. The entity's primary identifier / name is stored in `n.id`.
3. Document text chunks have label `Document` with properties: `id`, `text`, `scheme_title`, `department`.
4. Documents connect to entities via: `(d:Document)-[:MENTIONS]->(e:__Entity__)`.
5. Entities connect via relationships such as `BENEFITS`, `PROVIDES`, `SPONSORED_BY`, `MANAGED_BY`, `PURPOSE_FOR_LOAN`, `MAXIMUM_LOAN_AMOUNT`, etc.

Strict Query Rules:
- Return ONLY valid Cypher code. Do NOT wrap in markdown fences (no ```cypher).
- The query MUST be a read-only MATCH statement (NO CREATE, SET, MERGE, DELETE).
- Place all MATCH and OPTIONAL MATCH clauses BEFORE the final RETURN statement.
- Filter for relevant entities using exact IDs when available (e.g. `WHERE s.id IN [...]` or `WHERE toLower(s.id) CONTAINS ...`).
- OPTIONAL MATCH connected details: `OPTIONAL MATCH (s)-[r]->(target:__Entity__)` and `OPTIONAL MATCH (d:Document)-[:MENTIONS]->(s)`.
- RETURN informative columns, for example: `s.id AS scheme_name`, `type(r) AS relationship`, `target.id AS target_entity`, `d.text AS document_text`.
- Limit the returned records to 25 to avoid overwhelming context.
"""

    response = llm.invoke(prompt)
    cypher = response.content.strip()

    # Clean markdown fences if any were included
    if cypher.startswith("```"):
        lines = cypher.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cypher = "\n".join(lines).strip()

    return cypher


def execute_cypher_and_synthesize(
    question: str,
    cypher_query: str,
    matched_nodes: Optional[List[Dict]] = None,
    graph: Optional[Neo4jGraph] = None,
    llm: Optional[ChatOpenAI] = None,
) -> Dict:
    """
    Step 4: Execute the generated Cypher query in Neo4j and synthesize the final answer with LLM.
    """
    if graph is None:
        graph = _get_neo4j_graph()
    if llm is None:
        llm = _get_llm(temperature=0.0)

    query_results = []
    error_msg = None

    try:
        query_results = graph.query(cypher_query)
    except Exception as e:
        error_msg = str(e)

    # Fallback if query failed or returned empty results
    if (not query_results or error_msg) and matched_nodes:
        fallback_ids = [n["id"] for n in matched_nodes[:5] if n.get("id")]
        fallback_query = """
        UNWIND $ids AS nid
        MATCH (s:__Entity__ {id: nid})
        OPTIONAL MATCH (s)-[r]->(target:__Entity__)
        OPTIONAL MATCH (d:Document)-[:MENTIONS]->(s)
        RETURN s.id AS scheme_name, type(r) AS relationship, target.id AS target_entity, d.text AS document_text
        LIMIT 25
        """
        try:
            query_results = graph.query(fallback_query, {"ids": fallback_ids})
        except Exception:
            pass

    # Build context string for the synthesis LLM
    context_lines = []
    for r in query_results:
        parts = []
        if r.get("scheme_name"):
            parts.append(f"Entity/Scheme: {r['scheme_name']}")
        if r.get("relationship") and r.get("target_entity"):
            parts.append(f"Relationship: {r['relationship']} -> {r['target_entity']}")
        if r.get("document_text"):
            parts.append(f"Document Details: {r['document_text']}")
        if parts:
            context_lines.append(" | ".join(parts))

    graph_context = "\n\n".join(context_lines) if context_lines else "No direct matching graph data found."

    synthesis_prompt = f"""You are an intelligent AI assistant answering user questions using knowledge retrieved from a Neo4j Knowledge Graph about Government Schemes.

User Question:
{question}

Retrieved Knowledge Graph Context:
{graph_context}

Instructions:
1. Provide a comprehensive, clear, and well-structured answer explaining the relevant schemes, benefits, funding patterns, eligibility, and how they address the user question.
2. Base your response strictly on the retrieved knowledge graph context.
3. If specific scheme details (like department, subsidy percentage, loan amounts, or equipment) are present, highlight them clearly.
"""

    answer_res = llm.invoke(synthesis_prompt)

    return {
        "cypher_executed": cypher_query,
        "error": error_msg,
        "results_count": len(query_results),
        "raw_results": query_results,
        "final_answer": answer_res.content,
    }



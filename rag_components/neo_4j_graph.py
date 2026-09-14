import os
from typing import List
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument

load_dotenv()


def _get_neo4j_graph():
    url = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", username)
    return Neo4jGraph(url=url, username=username, password=password, database=database)

def add_entity_graphs(graph_documents: List[GraphDocument]):
    graph = _get_neo4j_graph()
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True,
    )

    return graph

def get_graph_for_query(query: str):
    graph = _get_neo4j_graph()
    query = """
    MATCH (s)
    WHERE toLower(s.id) CONTAINS toLower($query)

    OPTIONAL MATCH (s)-[r]->(target)

    RETURN
        s.id AS source,
        type(r) AS relationship,
        target.id AS target
    """
    return graph.query(query, params={"query": query})
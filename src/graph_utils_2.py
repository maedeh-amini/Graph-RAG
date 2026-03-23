
# this script is used later to return the esample_id of the 5% train examples used for building the database for
# for the sake of valid querying


from neo4j import GraphDatabase
from typing import List
import os
from dotenv import load_dotenv


def list_graph_documents(uri: str, username: str, password: str, database: str) -> List[dict]:
    """
    List all document nodes stored in the Neo4j graph database.
    Prints example_id, title, and content for each node.
    """

    driver = GraphDatabase.driver(uri, auth=(username, password))

    query = """
    MATCH (d)
    WHERE d.example_id IS NOT NULL 
      AND d.title IS NOT NULL 
      AND d.page_content IS NOT NULL
    RETURN d.example_id AS example_id,
           d.title AS title,
           d.page_content AS content
    ORDER BY example_id
    """

    documents = []

    with driver.session(database=database) as session:
        results = session.run(query)

        for record in results:
            documents.append({
                "example_id": record["example_id"],
                "title": record["title"],
                "content": record["content"]
            })

    driver.close()

    print(f"\n[INFO] Total documents in graph: {len(documents)}\n")

    for i, doc in enumerate(documents):
        print(f"--- Document {i} ---")
        print("example_id:", doc["example_id"])
        print("title:", doc["title"])
        print("content:", doc["content"][:200], "...")
        print()

    return documents


def list_graph_documents_by_example_ids(
    uri: str,
    username: str,
    password: str,
    database: str,
    example_ids: List[str]
) -> List[dict]:
    """
    List only documents whose 'example_id' is in the provided list.
    Useful for inspecting the subset stored in your graph.
    """

    driver = GraphDatabase.driver(uri, auth=(username, password))

    query = """
    MATCH (d)
    WHERE d.example_id IN $ids
    RETURN d.example_id AS example_id,
           d.title AS title,
           d.page_content AS content
    ORDER BY example_id
    """

    documents = []

    with driver.session(database=database) as session:
        results = session.run(query, ids=example_ids)

        for record in results:
            documents.append({
                "example_id": record["example_id"],
                "title": record["title"],
                "content": record["content"]
            })

    driver.close()

    print(f"\n[INFO] Total documents in graph (filtered): {len(documents)}\n")

    for i, doc in enumerate(documents):
        print(f"--- Document {i} ---")
        print("example_id:", doc["example_id"])
        print("title:", doc["title"])
        print("content:", doc["content"][:200], "...")
        print()

    return documents


# ---------------------------------------------------------
# Run this file directly to inspect the graph database
# ---------------------------------------------------------
if __name__ == "__main__":

    load_dotenv()

    NEO4J_URI = os.environ["NEO4J_URI"]
    NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
    NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
    NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]

    print("\n[INFO] Inspecting Neo4j graph documents...\n")

    docs = list_graph_documents(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )
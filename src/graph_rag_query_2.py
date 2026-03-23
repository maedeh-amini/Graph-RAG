


"""
graph_rag_query.py
------------------
Read-only querying of the Graph-RAG pipeline built in graph_rag_pipeline.py.

Requirements:
- The Neo4j database must already have the graph populated by graph_rag_pipeline.py
- graph_rag_pipeline.py must contain GraphRAGChain, create_cypher_prompt, graph, qa_llm, cypher_llm
"""

from src.graph_rag_pipeline_2 import GraphRAGChain, create_cypher_prompt, graph, qa_llm, cypher_llm

class GraphRAGQuery:
    """
    Provides a safe, read-only interface to query the existing Graph-RAG Neo4j graph.
    Does NOT modify the graph database.
    """

    def __init__(self, graph, qa_llm, cypher_llm):
        # Build a GraphCypherQAChain from the existing graph
        self.graph_rag_chain = GraphRAGChain(graph=graph, qa_llm=qa_llm, cypher_llm=cypher_llm)
        prompt_template = create_cypher_prompt()
        self.chain = self.graph_rag_chain.build_chain(prompt_template)

    def query(self, question: str) -> str:
        """
        Run a read-only query on the graph and return the answer.
        """
        if not self.chain:
            raise RuntimeError("Chain not initialized")
        response = self.chain.invoke({"query": question})
        return response


# ==========================
# Example usage
# ==========================
if __name__ == "__main__":
    # Initialize the query object
    query_engine = GraphRAGQuery(graph=graph, qa_llm=qa_llm, cypher_llm=cypher_llm)

    # Example question
    question = "What is Old Orchard Shopping Center currently called??"

    answer = query_engine.query(question)
    print("\n[QUERY RESULT]")
    print(answer)
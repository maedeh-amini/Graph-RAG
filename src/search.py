from langchain_neo4j import GraphCypherQAChain


class GraphQueryService:
    def __init__(self, graph, qa_llm, cypher_llm):
        self.chain = GraphCypherQAChain.from_llm(
            llm=qa_llm,
            cypher_llm=cypher_llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True,
        )

    def query(self, question: str) -> dict:
        return self.chain.invoke({"query": question})

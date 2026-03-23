from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer


class Neo4jGraphBuilder:
    def __init__(self, uri, username, password, database, llm):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.llm = llm

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
            max_connection_lifetime=3600,
        )

        self.graph = Neo4jGraph(
            url=self.uri,
            username=self.username,
            password=self.password,
            database=self.database,
            sanitize=True,
        )

    def test_connection(self):
        with self.driver.session(database=self.database) as session:
            result = session.run("RETURN 'connected' AS status")
            return result.single()

    def build_graph(self, documents):
        transformer = LLMGraphTransformer(llm=self.llm)
        graph_docs = transformer.convert_to_graph_documents(documents)
        self.graph.add_graph_documents(graph_docs)
        self.graph.refresh_schema()
        return graph_docs

    def get_graph(self):
        return self.graph

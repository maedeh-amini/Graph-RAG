from langchain_openai import ChatOpenAI
from config import AppConfig
from data_loader import PDFDataLoader
from graph_builder import Neo4jGraphBuilder
from search import GraphQueryService


def main():
    # LLMs
    cypher_llm = ChatOpenAI(
        api_key=AppConfig.ACADEMIC_API_KEY,
        base_url=AppConfig.BASE_URL,
        model=AppConfig.CYPHER_MODEL,
        temperature=0,
    )

    qa_llm = ChatOpenAI(
        api_key=AppConfig.ACADEMIC_API_KEY,
        base_url=AppConfig.BASE_URL,
        model=AppConfig.QA_MODEL,
        temperature=0,
    )

    # Load data
    loader = PDFDataLoader("data")
    documents = loader.load()

    # Build graph
    builder = Neo4jGraphBuilder(
        uri=AppConfig.NEO4J_URI,
        username=AppConfig.NEO4J_USERNAME,
        password=AppConfig.NEO4J_PASSWORD,
        database=AppConfig.NEO4J_DATABASE,
        llm=qa_llm,
    )

    print(builder.test_connection())
    builder.build_graph(documents)
    graph = builder.get_graph()

    # Query layer
    query_service = GraphQueryService(graph, qa_llm, cypher_llm)
    response = query_service.query("What is good learning??")
    print(response)


if __name__ == "__main__":
    main()

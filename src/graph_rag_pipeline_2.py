"""
Graph-RAG Pipeline for HotpotQA (RAGBench) - Modular Version
-------------------------------------------------------------
- Load a subset of HotpotQA train split.
- Convert documents into graph nodes using LLMGraphTransformer.
- Populate Neo4j graph database.
- Build GraphCypherQAChain for multi-hop queries.
- Design to be imported for query scripts.
"""

import os
import random
from tqdm import tqdm
from datasets import load_dataset
from neo4j import GraphDatabase
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# =========================================================
# 1. Load environment variables
# =========================================================
load_dotenv()
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
UOS_API_KEY = os.environ["UOS_API_KEY"]
UOS_API_BASE = os.environ["UOS_API_BASE"]

# =========================================================
# 2. Connect to Neo4j
# =========================================================
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    max_connection_lifetime=3600
)

with driver.session(database=NEO4J_DATABASE) as session:
    result = session.run("RETURN 'connected' AS status")
    print("Neo4j status:", result.single()["status"])

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
    sanitize=True
)

# =========================================================
# 3. LLM initialization
# =========================================================
cypher_llm = ChatOpenAI(api_key=UOS_API_KEY, base_url=UOS_API_BASE, model="gemma3", temperature=0.2)
qa_llm = ChatOpenAI(api_key=UOS_API_KEY, base_url=UOS_API_BASE, model="gemma3", temperature=0.2)

# =========================================================
# 4. Classes
# =========================================================

class GraphRAGBuilder:
    """
    Build Graph-RAG by loading HotpotQA subset, converting to graph documents,
    and populating Neo4j.
    """

    def __init__(self, graph: Neo4jGraph, llm: ChatOpenAI, subset_fraction: float = 0.05):
        self.graph = graph
        self.llm = llm
        self.subset_fraction = subset_fraction
        self.documents = []
        self.graph_documents = []

    def load_documents(self):
        """Load HotpotQA subset and convert to LangChain Documents."""
        dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="train")
        total_rows = len(dataset)
        subset_size = max(1, int(total_rows * self.subset_fraction))
        subset_indices = random.sample(range(total_rows), subset_size)

        doc_id = 0
        for idx in tqdm(subset_indices, desc="Converting dataset to Documents"):
            row = dataset[idx]
            for sentences in row["documents_sentences"]:
                text = " ".join(s_text for s_id, s_text in sentences if s_text and s_text.strip())
                if len(text.split()) < 20:
                    continue
                self.documents.append(Document(
                    page_content=text,
                    metadata={"doc_id": doc_id, "example_id": row["id"], "title": f"doc_{doc_id}"}
                ))
                doc_id += 1

        print(f"[INFO] Extracted {len(self.documents)} documents")
        return self.documents

    def convert_to_graph_documents(self):
        """Convert Documents into Graph Documents (nodes + relationships)."""
        transformer = LLMGraphTransformer(llm=self.llm)
        for doc in tqdm(self.documents, desc="Converting Documents to Graph Documents"):
            g_docs = transformer.convert_to_graph_documents([doc])
            self.graph_documents.extend(g_docs)
        return self.graph_documents

    def populate_graph(self):
        """Populate Neo4j and refresh schema."""
        self.graph.add_graph_documents(self.graph_documents)
        self.graph.refresh_schema()
        print("[INFO] Graph schema updated")
        return self.graph


class GraphRAGChain:
    """
    Encapsulates the GraphCypherQAChain.
    """

    def __init__(self, graph: Neo4jGraph, qa_llm: ChatOpenAI, cypher_llm: ChatOpenAI):
        self.graph = graph
        self.qa_llm = qa_llm
        self.cypher_llm = cypher_llm
        self.chain = None

    def build_chain(self, prompt_template: PromptTemplate):
        """Create GraphCypherQAChain for queries."""
        self.chain = GraphCypherQAChain.from_llm(
            llm=self.qa_llm,
            cypher_llm=self.cypher_llm,
            graph=self.graph,
            prompt=prompt_template,
            verbose=True,
            allow_dangerous_requests=True
        )
        return self.chain


def create_cypher_prompt():
    """Prompt template for Cypher generation."""
    template = """
You are a Neo4j Cypher expert for HotpotQA multi-hop QA.

TASK:
Generate ONE Cypher query that retrieves ALL properties
required to answer the question.

STRICT RULES:
- Output ONLY executable Cypher.
- No explanations, markdown, comments, or reasoning.
- Return scalar properties only with aliases.
- Use only labels, relationships, and properties in the schema.
- Never use CREATE, MERGE, DELETE, SET.
- Avoid OPTIONAL MATCH unless necessary.

ANSWER COMPLETENESS:
- Multi-part questions must retrieve all facts explicitly.
- If information is incomplete, retrieve everything available.
- Always produce a final answer; never return empty output.

SCHEMA:
{schema}

QUESTION:
{question}
"""
    return PromptTemplate(input_variables=["schema", "question"], template=template)


# =========================================================
# 5. Build pipeline if run as main
# =========================================================
if __name__ == "__main__":
    builder = GraphRAGBuilder(graph=graph, llm=qa_llm, subset_fraction=0.05)
    builder.load_documents()
    builder.convert_to_graph_documents()
    builder.populate_graph()

    prompt_template = create_cypher_prompt()
    chain_builder = GraphRAGChain(graph=graph, qa_llm=qa_llm, cypher_llm=cypher_llm)
    chain = chain_builder.build_chain(prompt_template)

    # Sample query
    query = "In what school district is Governor John R. Rogers High School located?"
    answer = chain.invoke({"query": query})
    print("\nSample Answer:")
    print(answer)
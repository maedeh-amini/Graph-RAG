"""
Full Graph-RAG Pipeline for HotpotQA (RAGBench) with Subset and Progress Bars
-------------------------------------------------------------------------------
This pipeline demonstrates a graph-based Retrieval-Augmented Generation (RAG)
system using the HotpotQA dataset from RAGBench. Only the 'documents_sentences'
column is used to build graph nodes.

Features:
- Random 20% subset of the HotpotQA train split (configurable)
- Conversion of dataset rows to LangChain Documents
- Conversion of Documents to Graph Documents using LLMGraphTransformer
- Progress bars for monitoring
- GraphCypherQAChain for multi-hop question answering on Neo4j
- Clear documentation for each stage

Requirements:
- Neo4j database configured with credentials in environment variables
- LangChain with Neo4j and OpenAI support
- datasets, tqdm, python-dotenv
"""

import os
import random
from neo4j import GraphDatabase
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from tqdm import tqdm

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
# Establish a Neo4j driver and verify connection
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    max_connection_lifetime=3600
)

with driver.session(database=NEO4J_DATABASE) as session:
    result = session.run("RETURN 'connected' AS status")
    print("Neo4j status:", result.single()["status"])

# Initialize LangChain Neo4jGraph object
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
    sanitize=True
)

# =========================================================
# 3. Initialize LLMs for Cypher generation and QA
# =========================================================
# ChatOpenAI LLMs will only be used at query time
CYPHER_MODEL = "gpt-oss"
QA_MODEL = "gpt-oss"

cypher_llm = ChatOpenAI(
    api_key=UOS_API_KEY,
    base_url=UOS_API_BASE,
    model=CYPHER_MODEL,
    temperature=0.2
)

qa_llm = ChatOpenAI(
    api_key=UOS_API_KEY,
    base_url=UOS_API_BASE,
    model=QA_MODEL,
    temperature=0.2
)

# =========================================================
# 4. Load HotpotQA dataset and build Document objects
# =========================================================

def load_subset_documents_graph_rag(subset_fraction: float = 0.05) -> list[Document]:
    """
    Load a random subset of HotpotQA train split and convert to LangChain Document objects.

    Parameters
    ----------
    subset_fraction : float
        Fraction of the train split to use (0 < subset_fraction <= 1)

    Returns
    -------
    documents : list[Document]
        List of LangChain Document objects, each representing a single passage
        flattened from the 'documents_sentences' column. Each document has metadata:
        - doc_id: unique integer id
        - example_id: original HotpotQA example id
        - title: generated title as 'doc_{doc_id}'
    """
    documents = []

    dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="train")
    total_rows = len(dataset)
    subset_size = max(1, int(total_rows * subset_fraction))
    print(f"[INFO] Total rows in train split: {total_rows}, using {subset_size} rows (~{subset_fraction*100}%)")

    # randomly select subset indices
    subset_indices = random.sample(range(total_rows), subset_size)
    
    doc_id = 0
    # Loop with progress bar
    for idx in tqdm(subset_indices, desc="Converting dataset to Documents"):
        row = dataset[idx]
        docs_sentences = row["documents_sentences"]
        for sentences in docs_sentences:
            text = " ".join(s_text for s_id, s_text in sentences if s_text and s_text.strip())
            if len(text.split()) < 20:  # Skip very short passages
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "doc_id": doc_id,
                        "example_id": row["id"],
                        "title": f"doc_{doc_id}"
                    }
                )
            )
            doc_id += 1

    print(f"[INFO] Extracted {len(documents)} Document objects for Graph-RAG subset")
    return documents

documents = load_subset_documents_graph_rag(subset_fraction=0.05)

# =========================================================
# 5. Convert Documents into Graph Documents
# =========================================================
# LLMGraphTransformer converts raw text into structured nodes + edges for the graph

llm_transformer = LLMGraphTransformer(llm=qa_llm)

graph_documents = []
for doc in tqdm(documents, desc="Converting Documents to Graph Documents"):
    g_docs = llm_transformer.convert_to_graph_documents([doc])
    graph_documents.extend(g_docs)

# Add graph documents to Neo4j and refresh schema
graph.add_graph_documents(graph_documents)
graph.refresh_schema()
print("[INFO] Graph schema updated:")
print(graph.schema)

# =========================================================
# 6. Define Cypher generation prompt
# =========================================================
CYPHER_GENERATION_TEMPLATE = """
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

prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)

# =========================================================
# 7. Build GraphCypherQAChain
# =========================================================
chain = GraphCypherQAChain.from_llm(
    llm=qa_llm,
    cypher_llm=cypher_llm,
    graph=graph,
    prompt=prompt,
    verbose=True,
    allow_dangerous_requests=True
)

# =========================================================
# 8. Sample query execution
# =========================================================
query = "In what school district is Governor John R. Rogers High School, named after John Rankin Rogers, located?"

response = chain.invoke({"query": query})

print("\nFinal Answer:")
print(response)





















#-----------------------
#  Old Verion 2
#-----------------------

# """
# Full Graph-RAG Pipeline for HotpotQA (RAGBench)
# ------------------------------------------------
# Uses only 'documents_sentences' column for graph construction.
# """

# import os
# from neo4j import GraphDatabase
# from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
# from langchain_openai import ChatOpenAI
# from datasets import load_dataset
# from langchain_core.documents import Document
# from langchain_experimental.graph_transformers import LLMGraphTransformer
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv
# load_dotenv()

# # =========================================================
# # 1. Environment variables
# # =========================================================

# NEO4J_URI = os.environ["NEO4J_URI"]
# NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
# NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
# NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
# UOS_API_KEY = os.environ["UOS_API_KEY"]
# UOS_API_BASE=os.environ["UOS_API_BASE"]

# # =========================================================
# # 2. Neo4j connection
# # =========================================================

# driver = GraphDatabase.driver(
#     NEO4J_URI,
#     auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
#     max_connection_lifetime=3600
# )

# with driver.session(database=NEO4J_DATABASE) as session:
#     result = session.run("RETURN 'connected' AS status")
#     print("Neo4j status:", result.single()["status"])

# graph = Neo4jGraph(
#     url=NEO4J_URI,
#     username=NEO4J_USERNAME,
#     password=NEO4J_PASSWORD,
#     database=NEO4J_DATABASE,
#     sanitize=True
# )

# # =========================================================
# # 3. Initialize LLMs
# # =========================================================

# CYPHER_MODEL = "gpt-oss"
# QA_MODEL = "gpt-oss"

# cypher_llm = ChatOpenAI(
#     api_key=UOS_API_KEY,
#     base_url=UOS_API_BASE,
#     model=CYPHER_MODEL,
#     temperature=0.2
# )

# qa_llm = ChatOpenAI(
#     api_key=UOS_API_KEY,
#     base_url=UOS_API_BASE,
#     model=QA_MODEL,
#     temperature=0.2
# )

# # =========================================================
# # 4. Load HotpotQA dataset and build Documents
# # =========================================================

# def load_all_documents_graph_rag() -> list[Document]:
#     """
#     Load HotpotQA train split and convert 'documents_sentences' into Document objects.

#     Each document is flattened into a single passage. Titles are generated as 'doc_{doc_id}'.
#     """
#     documents = []

#     dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="train")
#     print(f"[DEBUG] Loaded HotpotQA train split: {len(dataset)} rows")

#     dataset = dataset.remove_columns(
#         [col for col in dataset.column_names if col != "documents_sentences" and col != "id"]
#     )

#     doc_id = 0
#     for row in dataset:
#         docs_sentences = row["documents_sentences"]
#         for sentences in docs_sentences:
#             text = " ".join(s_text for s_id, s_text in sentences if s_text and s_text.strip())
#             if len(text.split()) < 20:
#                 continue

#             documents.append(
#                 Document(
#                     page_content=text,
#                     metadata={
#                         "doc_id": doc_id,
#                         "example_id": row["id"],
#                         "title": f"doc_{doc_id}"
#                     }
#                 )
#             )
#             doc_id += 1

#     print(f"[DEBUG] Extracted {len(documents)} Document objects for Graph-RAG")
#     return documents

# documents = load_all_documents_graph_rag()

# # =========================================================
# # 5. Convert Documents into Graph Documents
# # =========================================================

# llm_transformer = LLMGraphTransformer(llm=qa_llm)
# graph_documents = llm_transformer.convert_to_graph_documents(documents)

# graph.add_graph_documents(graph_documents)
# graph.refresh_schema()
# print("Graph schema updated:")
# print(graph.schema)

# # =========================================================
# # 6. Define Cypher generation prompt
# # =========================================================

# CYPHER_GENERATION_TEMPLATE = """
# You are a Neo4j Cypher expert for HotpotQA multi-hop QA.

# TASK:
# Generate ONE Cypher query that retrieves ALL properties
# required to answer the question.

# STRICT RULES:
# - Output ONLY executable Cypher.
# - No explanations, markdown, comments, or reasoning.
# - Return scalar properties only with aliases.
# - Use only labels, relationships, and properties in the schema.
# - Never use CREATE, MERGE, DELETE, SET.
# - Avoid OPTIONAL MATCH unless necessary.

# ANSWER COMPLETENESS:
# - Multi-part questions must retrieve all facts explicitly.
# - If information is incomplete, retrieve everything available.
# - Always produce a final answer; never return empty output.

# SCHEMA:
# {schema}

# QUESTION:
# {question}
# """

# prompt = PromptTemplate(
#     input_variables=["schema", "question"],
#     template=CYPHER_GENERATION_TEMPLATE,
# )

# # =========================================================
# # 7. Build GraphCypherQAChain
# # =========================================================

# chain = GraphCypherQAChain.from_llm(
#     llm=qa_llm,
#     cypher_llm=cypher_llm,
#     graph=graph,
#     prompt=prompt,
#     verbose=True,
#     allow_dangerous_requests=True
# )

# # =========================================================
# # 8. Sample query execution
# # =========================================================

# query = "In what school district is Governor John R. Rogers High School, named after John Rankin Rogers, located?"

# response = chain.invoke({"query": query})

# print("\nFinal Answer:")
# print(response)











#-----------------------
#  Old Verion 1
#-----------------------

# import os
# from dotenv import load_dotenv
# from neo4j import GraphDatabase
# from langchain_neo4j import Neo4jGraph
# # from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
# from datasets import load_dataset
# from langchain_core.documents import Document
# from langchain_experimental.graph_transformers import LLMGraphTransformer
# import pandas as pd
# from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
# from langchain_experimental.graph_transformers import LLMGraphTransformer
# from langchain_neo4j import GraphCypherQAChain
# from langchain_community.document_loaders import PyPDFDirectoryLoader

# load_dotenv()
# NEO4J_URI=os.environ["NEO4J_URI"]
# NEO4J_USERNAME=os.environ["NEO4J_USERNAME"]
# NEO4J_PASSWORD=os.environ["NEO4J_PASSWORD"]
# NEO4J_DATABASE=os.environ["NEO4J_DATABASE"]
# ACADEMIC_API_KEY=os.environ["ACADEMIC_API_KEY"]



# # # -------------------------
# # # Test connection to Neo4j
# # # -------------------------
# driver = GraphDatabase.driver(
#     NEO4J_URI,
#     auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
#     max_connection_lifetime=3600 
# )

# with driver.session(database=NEO4J_DATABASE) as session:
#     result = session.run("RETURN 'connected' AS status")
#     print(result.single())

# # driver.close()



# # # -------------------------
# # # Initialize LangChain Neo4j wrapper
# # # -------------------------
# graph=Neo4jGraph(
#     url=NEO4J_URI,
#     username=NEO4J_USERNAME,
#     password=NEO4J_PASSWORD,
#     database=NEO4J_DATABASE, 
#     sanitize=True # Cypher queries generated are safe to execute
# )


# graph # This will just display the object in interactive environments


# # # -------------------------
# # # Initialize OpenAI LLM
# # # -------------------------

# CYPHER_MODEL="gpt-oss-120b"
# QA_MODEL="gpt-oss-120b"

# cypher_llm = ChatOpenAI(
#     api_key=ACADEMIC_API_KEY,  
#     base_url="https://chat-ai.academiccloud.de/v1",
#     model=CYPHER_MODEL,
#     temperature=0
# )

# qa_llm=ChatOpenAI(
#     api_key=ACADEMIC_API_KEY,  
#     base_url="https://chat-ai.academiccloud.de/v1",
#     model=QA_MODEL,
#     temperature=0
# )

# # response=qa_llm.invoke("How tall is the Eiffel tower?")


# # # -------------------------
# # # Load data
# # # -------------------------
# loader = PyPDFDirectoryLoader("data")   
# docs = loader.load()

# print(f"Loaded {len(docs)} document pages")
# print(docs[0].page_content[:500])  # preview first 500 chars
# documents= [Document(page_content=str(docs))]

# # # -------------------------
# # # Transform documents to graph format
# # # -------------------------
# llm_transformer=LLMGraphTransformer(llm=qa_llm)
# graph_documents=llm_transformer.convert_to_graph_documents(documents) #convert raw text to structured knowledge--> nodes + relationships

# # # Inspect first graph document
# print(graph_documents)
# print(graph_documents[0].nodes)
# print(graph_documents[0].relationships)



# # # -------------------------
# # # Add documents to Neo4j via LangChain wrapper
# # # -------------------------
# graph.add_graph_documents(graph_documents)

# # # -------------------------
# # # Refresh graph schema and inspect
# # # -------------------------
# graph.refresh_schema()
# print(graph.schema)


# # # -------------------------
# # # Create the Cypher QA chain from your LLM and existing graph
# # # -------------------------


# chain = GraphCypherQAChain.from_llm(
# 	llm=qa_llm,
# 	cypher_llm=cypher_llm,
# 	graph=graph,
#     # cypher_prompt=cypher_prompt,
# 	verbose=True,
#     allow_dangerous_requests=True
# )
# print(chain)

# # # Run a natural language query
# # response=chain.invoke({"query":"How to cancel an order"})
# # print(response)





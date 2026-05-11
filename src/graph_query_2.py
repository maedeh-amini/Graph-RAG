import os
from typing import List
from dotenv import load_dotenv

from langchain_neo4j import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load configuration
load_dotenv()

# =========================================================
# 2. Custom Embedding Provider
# =========================================================
class GatewayCompatibleEmbeddings(OpenAIEmbeddings):
    """
    A professional wrapper to ensure compatibility with LiteLLM gateways 
    by stripping unsupported encoding parameters from the API call.
    """
    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str = None, **kwargs
    ) -> List[List[float]]:
        # Remove parameters often rejected by non-standard OpenAI gateways
        kwargs.pop("encoding_format", None)
        kwargs.pop("chunk_size", None)
        
        responses = self.client.create(
            input=texts,
            model=self.model,
            encoding_format=None, 
            **kwargs
        )
        return [d.embedding for d in responses.data]

# =========================================================
# 3. Component Initialization
# =========================================================

# Initialize the refined embedding provider
embeddings = GatewayCompatibleEmbeddings(
    api_key=os.environ["EMBEDDING_MODEL_API_KEY"],
    base_url=os.environ["EMBEDDING_MODEL_API_BASE"],
    model="bge-m3"
)

# Initialize the LLM 
llm = ChatOpenAI(
    api_key=os.environ["UOS_API_KEY"],
    base_url=os.environ["UOS_API_BASE"],
    model="gpt-oss",
    temperature=0
)

# =========================================================
# 4. Graph Retrieval Strategy
# =========================================================

# This query finds relevant text chunks via vector search, 
# then traverses to find related entities and their neighbors.
retrieval_query = """
MATCH (node)-[:HAS_ENTITY|MENTIONS]->(e)
OPTIONAL MATCH (e)-[r]->(neighbor)
WITH node, score, 
     collect(DISTINCT e.id) as entities, 
     collect(DISTINCT type(r) + ' -> ' + neighbor.id) as relations
RETURN 
    node.text + "\\n\\nGRAPH CONTEXT:\\nEntities: " + apoc.text.join(entities, ", ") + 
    "\\nRelationships: " + apoc.text.join(relations, ", ") AS text,
    score, 
    {source: node.source} AS metadata
"""

# Establish connection specifically to the 'test01' database
vector_db = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    database=os.environ.get("NEO4J_DATABASE", "graphdb"), 
    index_name="entity_vector_index", 
    search_type="vector",  # Explicitly use the vector search engine
    retrieval_query=retrieval_query
)

# =========================================================
# 5. RAG Chain Construction (LCEL)
# =========================================================

template = """Answer the question based only on the provided context, 
which includes both raw text and structural graph relationships:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    """Aggregates retrieved documents into a context block."""
    return "\n\n".join(doc.page_content for doc in docs)

# Define the pipeline
# k=3 retrieves the top 3 most relevant segments from the graph
rag_chain = (
    {"context": vector_db.as_retriever(search_kwargs={'k': 5}) | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# =========================================================
# 6. Execution
# =========================================================
if __name__ == "__main__":
    question = "What was the nickname of Anthony Corallo, boss to 'Sal' Avellino and head of the Lucchese mob family mob in New York?"
    
    print(f"\n[QUERYING SYSTEM]: {question}")
    try:
        response = rag_chain.invoke(question)
        print(f"\n[SYSTEM RESPONSE]:\n{response}")
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}")







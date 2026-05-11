
# import os
# import json
# import random
# from typing import List
# from tqdm import tqdm
# from dotenv import load_dotenv
# from datasets import load_dataset
# from neo4j import GraphDatabase

# import deepeval
# from deepeval.test_case import LLMTestCase

# from deepeval.metrics import (
#     AnswerRelevancyMetric,
#     FaithfulnessMetric,
#     ContextualPrecisionMetric,
#     ContextualRecallMetric,
#     ContextualRelevancyMetric,
# )
# from deepeval.models.base_model import DeepEvalBaseLLM
# from langchain_neo4j import Neo4jVector
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate

# # 1. Load configuration
# load_dotenv()

# # --- DeepEval Global Safety Settings ---
# # These prevent the asyncio and TimeoutErrors for large datasets
# os.environ["DEEPEVAL_DISABLE_CONFIDENT"] = "false"
# os.environ["DEEPEVAL_ASYNC_MODE"] = "True"
# os.environ["DEEPEVAL_MAX_CONCURRENCY"] = "2"
# os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "1200"

# # =========================================================
# # 2. Component Initialization
# # =========================================================
# class GatewayCompatibleEmbeddings(OpenAIEmbeddings):
#     def _get_len_safe_embeddings(self, texts: List[str], *, engine: str = None, **kwargs) -> List[List[float]]:
#         kwargs.pop("encoding_format", None)
#         kwargs.pop("chunk_size", None)
#         responses = self.client.create(input=texts, model=self.model, encoding_format=None, **kwargs)
#         return [d.embedding for d in responses.data]

# embeddings = GatewayCompatibleEmbeddings(
#     api_key=os.environ["EMBEDDING_MODEL_API_KEY"],
#     base_url=os.environ["EMBEDDING_MODEL_API_BASE"],
#     model="bge-m3"
# )

# llm = ChatOpenAI(
#     api_key=os.environ["UOS_API_KEY"],
#     base_url=os.environ["UOS_API_BASE"],
#     model="gpt-oss",
#     temperature=0
# )

# # =========================================================
# # 3. Chunk-First Graph Traversal Logic
# # =========================================================
# retrieval_query = """
# OPTIONAL MATCH (node)-[:HAS_ENTITY|MENTIONS|RELATED]->(e)
# OPTIONAL MATCH (e)-[r]->(neighbor)
# WITH node, score, 
#      collect(DISTINCT coalesce(e.name, e.id, labels(e)[0])) as entities, 
#      collect(DISTINCT type(r) + ' -> ' + coalesce(neighbor.name, neighbor.id, labels(neighbor)[0])) as relations
# RETURN 
#     node.text + "\\n\\nGRAPH ENRICHMENT:\\nEntities: " + apoc.text.join(entities, ", ") + 
#     "\\nRelationships: " + apoc.text.join(relations, ", ") AS text,
#     score, 
#     {source: node.source, hf_id: node.hf_id} AS metadata
# """

# vector_db = Neo4jVector.from_existing_index(
#     embedding=embeddings,
#     url=os.environ["NEO4J_URI"],
#     username=os.environ["NEO4J_USERNAME"],
#     password=os.environ["NEO4J_PASSWORD"],
#     database=os.environ.get("NEO4J_DATABASE", "graphdb"), 
#     index_name="chunk_vector_index", 
#     search_type="vector",
#     retrieval_query=retrieval_query
# )

# # =========================================================
# # 4. RAG Pipeline Construction
# # =========================================================
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# template_str = "Answer the question based only on the provided context:\n{context}\n\nQuestion: {question}\nAnswer:"
# prompt = ChatPromptTemplate.from_template(template_str)

# rag_chain = (
#     {"context": vector_db.as_retriever(search_kwargs={'k': 5}) | format_docs, "question": RunnablePassthrough()}
#     | prompt | llm | StrOutputParser()
# )

# # =========================================================
# # 5. Data Handling
# # =========================================================
# RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# os.makedirs(RESULTS_DIR, exist_ok=True)

# def get_synchronized_dataset(sample_ratio=0.01):
#     driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
#     db = os.environ.get("NEO4J_DATABASE", "graphdb")
#     query = "MATCH (c:Chunk) WHERE c.hf_id IS NOT NULL RETURN c.hf_id AS id"
#     with driver.session(database=db) as session:
#         result = session.run(query)
#         graph_ids = {str(record["id"]) for record in result if record["id"]}
#     driver.close()
    
#     dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
#     filtered_data = [row for row in dataset if str(row["id"]) in graph_ids]
#     return random.sample(filtered_data, max(1, int(len(filtered_data) * sample_ratio)))

# def build_test_cases(dataset):
#     test_cases = []
#     retriever = vector_db.as_retriever(search_kwargs={'k': 5})

#     for row in tqdm(dataset, desc="Generating Test Cases"):
#         try:
#             docs = retriever.invoke(row["question"])
#             retrieved_texts = [doc.page_content for doc in docs]
#             answer = rag_chain.invoke(row["question"])
            
#             ground_truth_context = row["documents"]
#             if isinstance(ground_truth_context, str):
#                 ground_truth_context = [ground_truth_context]
            
#             test_cases.append(LLMTestCase(
#                 input=row["question"],
#                 expected_output=row["response"],
#                 actual_output=answer,
#                 retrieval_context=retrieved_texts,
#                 context=ground_truth_context
#             ))
#         except Exception as e:
#             print(f"\n[SKIP] ID {row['id']} failed: {e}")
#             continue
#     return test_cases

# # =========================================================
# # 6. Main Execution & Dynamic Summary
# # =========================================================
# class GPTOSSJudge(DeepEvalBaseLLM):
#     def __init__(self, model): self.model = model
#     def load_model(self): return self.model
#     def get_model_name(self): return "gpt-oss"
#     def generate(self, prompt: str) -> str: return self.model.invoke(prompt).content
#     async def a_generate(self, prompt: str) -> str: return self.generate(prompt)

# if __name__ == "__main__":
#     test_data = get_synchronized_dataset(sample_ratio=0.01) 
#     test_cases = build_test_cases(test_data)
    
#     if test_cases:
#         judge = GPTOSSJudge(llm)
#         metrics = [
#             ContextualRecallMetric(model=judge, threshold=0.7),
#             # ContextualRelevancyMetric(model=judge, threshold=0.7),
#             # FaithfulnessMetric(model=judge, threshold=0.7)
#         ]   
#         active_metric_names = ", ".join([m.__class__.__name__.replace("Metric", "") for m in metrics])
        
#         h_params = {
#             "model": "gpt-oss",
#             "temperature": 0,
#             "retrieval_k": 5,
#             "index_name": "chunk_vector_index",
#             "prompt_template": template_str
#         }

#         # 1. Run Evaluation with Batching to avoid internal framework hangs
#         batch_size = 20
#         all_test_results = []
        
#         print(f"\n[INFO] Starting Batched Evaluation ({len(test_cases)} cases)...")

#         for i in range(0, len(test_cases), batch_size):
#             batch = test_cases[i : i + batch_size]
#             print(f"\n>>> Processing Batch {(i // batch_size) + 1}...")
            
#             try:
#                 results_container = deepeval.evaluate(
#                     test_cases=batch, 
#                     metrics=metrics,
#                     hyperparameters=h_params
#                 )
#                 # Ensure we capture the list of results correctly
#                 if hasattr(results_container, 'test_results'):
#                     all_test_results.extend(results_container.test_results)
#                 else:
#                     all_test_results.extend(results_container)
#             except Exception as e:
#                 print(f"[BATCH ERROR] Failed at index {i}: {e}")
#                 continue

#         # 2. Extract results for clean JSON output
#         final_results_file = os.path.join(RESULTS_DIR, "final_results.json")
#         ordered_results = []
#         metric_stats = {}
#         total_tests = len(all_test_results)
#         total_passed_cases = 0

#         for result in all_test_results:
#             case_passed = result.success
#             if case_passed:
#                 total_passed_cases += 1

#             for m_data in result.metrics_data:
#                 m_name = m_data.name
#                 if m_name not in metric_stats:
#                     metric_stats[m_name] = {"passed": 0, "total": 0}
                
#                 metric_stats[m_name]["total"] += 1
#                 if m_data.success:
#                     metric_stats[m_name]["passed"] += 1

#             # CLEAN INTEGRATED MAPPING: Only includes the 7 requested fields
#             ordered_results.append({
#                 "status": "PASS" if case_passed else "FAIL",
#                 "input": result.input,
#                 "actual_output": result.actual_output,
#                 "expected_output": result.expected_output,
#                 "retrieval_context": result.retrieval_context,
#                 "ground_truth_context": result.context,
#                 "success": case_passed
#             })

#         # 4. Save Detailed Results
#         with open(final_results_file, "w", encoding="utf-8") as f:
#             json.dump(ordered_results, f, indent=2, ensure_ascii=False)

#         # 5. Save Summary
#         summary_file = os.path.join(RESULTS_DIR, "evaluation_summary.json")
        
#         summary_metrics = {
#             "overall_pass_rate": f"{(total_passed_cases / total_tests) * 100:.2f}%" if total_tests > 0 else "0.00%",
#             "total_test_cases": total_tests,
#             "passed_test_cases": total_passed_cases,
#             "failed_test_cases": total_tests - total_passed_cases,
#         }

#         for name, stats in metric_stats.items():
#             rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
#             summary_metrics[f"{name}_pass_rate"] = f"{rate:.2f}%"

#         summary_data = {
#             "summary_banner": f"Overall Metric Pass Rates - {active_metric_names}",
#             "hyperparameters": h_params,
#             "overall_metrics": summary_metrics,
#             "detailed_results_path": final_results_file
#         }

#         with open(summary_file, "w", encoding="utf-8") as f:
#             json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
#         print(f"\n[INFO] Summary saved to: {summary_file}")
#     else:
#         print("[ERROR] No cases generated.")







##################################################################################################
###### This is the version which works for contextual precision/ recall and answer relevancy #####
#####       However for Faithfulness and Contextual Relevancy it crashes                     #####
#####  It saves the summary and final results JSON files though                              #####
##################################################################################################


import os
import json
import random
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from neo4j import GraphDatabase

import deepeval
from deepeval.test_case import LLMTestCase

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_neo4j import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 1. Load configuration
load_dotenv()

# =========================================================
# 2. Component Initialization
# =========================================================
class GatewayCompatibleEmbeddings(OpenAIEmbeddings):
    def _get_len_safe_embeddings(self, texts: List[str], *, engine: str = None, **kwargs) -> List[List[float]]:
        kwargs.pop("encoding_format", None)
        kwargs.pop("chunk_size", None)
        responses = self.client.create(input=texts, model=self.model, encoding_format=None, **kwargs)
        return [d.embedding for d in responses.data]

embeddings = GatewayCompatibleEmbeddings(
    api_key=os.environ["EMBEDDING_MODEL_API_KEY"],
    base_url=os.environ["EMBEDDING_MODEL_API_BASE"],
    model="bge-m3"
)

llm = ChatOpenAI(
    api_key=os.environ["UOS_API_KEY"],
    base_url=os.environ["UOS_API_BASE"],
    model="gpt-oss",
    temperature=0
)

# =========================================================
# 3. Chunk-First Graph Traversal Logic
# =========================================================
retrieval_query = """
OPTIONAL MATCH (node)-[:HAS_ENTITY|MENTIONS|RELATED]->(e)
OPTIONAL MATCH (e)-[r]->(neighbor)
WITH node, score, 
     collect(DISTINCT coalesce(e.name, e.id, labels(e)[0])) as entities, 
     collect(DISTINCT type(r) + ' -> ' + coalesce(neighbor.name, neighbor.id, labels(neighbor)[0])) as relations
RETURN 
    node.text + "\\n\\nGRAPH ENRICHMENT:\\nEntities: " + apoc.text.join(entities, ", ") + 
    "\\nRelationships: " + apoc.text.join(relations, ", ") AS text,
    score, 
    {source: node.source, hf_id: node.hf_id} AS metadata
"""

vector_db = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    database=os.environ.get("NEO4J_DATABASE", "graphdb"), 
    index_name="chunk_vector_index", 
    search_type="vector",
    retrieval_query=retrieval_query
)

# =========================================================
# 4. RAG Pipeline Construction
# =========================================================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template_str = "Answer the question based only on the provided context:\n{context}\n\nQuestion: {question}\nAnswer:"
prompt = ChatPromptTemplate.from_template(template_str)

rag_chain = (
    {"context": vector_db.as_retriever(search_kwargs={'k': 5}) | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# =========================================================
# 5. Data Handling
# =========================================================
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_synchronized_dataset(sample_ratio=0.01):
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    db = os.environ.get("NEO4J_DATABASE", "graphdb")
    query = "MATCH (c:Chunk) WHERE c.hf_id IS NOT NULL RETURN c.hf_id AS id"
    with driver.session(database=db) as session:
        result = session.run(query)
        graph_ids = {str(record["id"]) for record in result if record["id"]}
    driver.close()
    
    dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
    filtered_data = [row for row in dataset if str(row["id"]) in graph_ids]
    return random.sample(filtered_data, max(1, int(len(filtered_data) * sample_ratio)))

def build_test_cases(dataset):
    test_cases = []
    retriever = vector_db.as_retriever(search_kwargs={'k': 5})

    for row in tqdm(dataset, desc="Generating Test Cases"):
        try:
            docs = retriever.invoke(row["question"])
            retrieved_texts = [doc.page_content for doc in docs]
            answer = rag_chain.invoke(row["question"])
            
            ground_truth_context = row["documents"]
            if isinstance(ground_truth_context, str):
                ground_truth_context = [ground_truth_context]
            
            test_cases.append(LLMTestCase(
                input=row["question"],
                expected_output=row["response"],
                actual_output=answer,
                retrieval_context=retrieved_texts,
                context=ground_truth_context
            ))
        except Exception as e:
            print(f"\n[SKIP] ID {row['id']} failed: {e}")
            continue
    return test_cases

# =========================================================
# 6. Main Execution & Dynamic Summary
# =========================================================
class GPTOSSJudge(DeepEvalBaseLLM):
    def __init__(self, model): self.model = model
    def load_model(self): return self.model
    def get_model_name(self): return "gpt-oss"
    def generate(self, prompt: str) -> str: return self.model.invoke(prompt).content
    async def a_generate(self, prompt: str) -> str: return self.generate(prompt)

if __name__ == "__main__":
    test_data = get_synchronized_dataset(sample_ratio=0.01) 
    test_cases = build_test_cases(test_data)
    
    if test_cases:
        judge = GPTOSSJudge(llm)
        metrics = [
        #   AnswerRelevancyMetric(model=judge, threshold=0.7),
        #   FaithfulnessMetric(model=judge, threshold=0.7),
        #   ContextualPrecisionMetric(model=judge, threshold=0.7),
            ContextualRecallMetric(model=judge, threshold=0.7),
        #   ContextualRelevancyMetric(model=judge, threshold=0.7)   #include_reason=True
        ]   
        active_metric_names = ", ".join([m.__class__.__name__.replace("Metric", "") for m in metrics])
        
        # Define Hyperparameters
        h_params = {
            "model": "gpt-oss",
            "temperature": 0,
            "retrieval_k": 5,
            "index_name": "chunk_vector_index",
            "prompt_template": template_str
        }

        # 1. Run Evaluation
        results_container = deepeval.evaluate(
            test_cases=test_cases, 
            metrics=metrics,
            hyperparameters=h_params
        )

        # 2. Extract results
        test_results_list = results_container.test_results
        final_results_file = os.path.join(RESULTS_DIR, "final_results.json")
        ordered_results = []
        metric_stats = {}
        total_tests = len(test_results_list)
        total_passed_cases = 0

        for result in test_results_list:
            case_passed = result.success
            if case_passed:
                total_passed_cases += 1

            for m_data in result.metrics_data:
                m_name = m_data.name
                if m_name not in metric_stats:
                    metric_stats[m_name] = {"passed": 0, "total": 0}
                
                metric_stats[m_name]["total"] += 1
                if m_data.success:
                    metric_stats[m_name]["passed"] += 1

            ordered_results.append({
                "status": "PASS" if case_passed else "FAIL",
                "input": result.input,
                "actual_output": result.actual_output,
                "expected_output": result.expected_output,
                "retrieval_context": result.retrieval_context,
                "ground_truth_context": result.context,
                "success": case_passed
            })

        # 4. Save Detailed Results
        with open(final_results_file, "w", encoding="utf-8") as f:
            json.dump(ordered_results, f, indent=2, ensure_ascii=False, default=str)

        # 5. Save Summary with Hyperparameters included
        summary_file = os.path.join(RESULTS_DIR, "evaluation_summary.json")
        
        summary_metrics = {
            "overall_pass_rate": f"{(total_passed_cases / total_tests) * 100:.2f}%" if total_tests > 0 else "0.00%",
            "total_test_cases": total_tests,
            "passed_test_cases": total_passed_cases,
            "failed_test_cases": total_tests - total_passed_cases,
        }

        for name, stats in metric_stats.items():
            rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            summary_metrics[f"{name}_pass_rate"] = f"{rate:.2f}%"

        summary_data = {
            "summary_banner": f"Overall Metric Pass Rates - {active_metric_names}",
            "hyperparameters": h_params,  # <--- Added your request here
            "overall_metrics": summary_metrics,
            "detailed_results_path": final_results_file
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
        print(f"\n[INFO] Summary saved to: {summary_file}")
    else:
        print("[ERROR] No cases generated.")











































# import os
# import json
# import random
# from typing import List
# from tqdm import tqdm
# from dotenv import load_dotenv
# from datasets import load_dataset
# from neo4j import GraphDatabase

# import deepeval
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import (
#     ContextualPrecisionMetric, 
#     ContextualRecallMetric, 
#     ContextualRelevancyMetric, 
#     AnswerRelevancyMetric, 
#     FaithfulnessMetric
# )

# from deepeval.models.base_model import DeepEvalBaseLLM

# from langchain_neo4j import Neo4jVector
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate

# # 1. Load configuration
# load_dotenv()

# # =========================================================
# # 2. Component Initialization
# # =========================================================
# class GatewayCompatibleEmbeddings(OpenAIEmbeddings):
#     def _get_len_safe_embeddings(self, texts: List[str], *, engine: str = None, **kwargs) -> List[List[float]]:
#         kwargs.pop("encoding_format", None)
#         kwargs.pop("chunk_size", None)
#         responses = self.client.create(input=texts, model=self.model, encoding_format=None, **kwargs)
#         return [d.embedding for d in responses.data]

# embeddings = GatewayCompatibleEmbeddings(
#     api_key=os.environ["EMBEDDING_MODEL_API_KEY"],
#     base_url=os.environ["EMBEDDING_MODEL_API_BASE"],
#     model="bge-m3"
# )

# llm = ChatOpenAI(
#     api_key=os.environ["UOS_API_KEY"],
#     base_url=os.environ["UOS_API_BASE"],
#     model="gpt-oss",
#     temperature=0
# )

# # =========================================================
# # 3. Chunk-First Graph Traversal Logic
# # =========================================================
# retrieval_query = """
# OPTIONAL MATCH (node)-[:HAS_ENTITY|MENTIONS|RELATED]->(e)
# OPTIONAL MATCH (e)-[r]->(neighbor)
# WITH node, score, 
#      collect(DISTINCT coalesce(e.name, e.id, labels(e)[0])) as entities, 
#      collect(DISTINCT type(r) + ' -> ' + coalesce(neighbor.name, neighbor.id, labels(neighbor)[0])) as relations
# RETURN 
#     node.text + "\\n\\nGRAPH ENRICHMENT:\\nEntities: " + apoc.text.join(entities, ", ") + 
#     "\\nRelationships: " + apoc.text.join(relations, ", ") AS text,
#     score, 
#     {source: node.source, hf_id: node.hf_id} AS metadata
# """

# vector_db = Neo4jVector.from_existing_index(
#     embedding=embeddings,
#     url=os.environ["NEO4J_URI"],
#     username=os.environ["NEO4J_USERNAME"],
#     password=os.environ["NEO4J_PASSWORD"],
#     database=os.environ.get("NEO4J_DATABASE", "graphdb"), 
#     index_name="chunk_vector_index", 
#     search_type="vector",
#     retrieval_query=retrieval_query
# )

# # =========================================================
# # 4. RAG Pipeline Construction
# # =========================================================
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# template_str = "Answer the question based only on the provided context:\n{context}\n\nQuestion: {question}\nAnswer:"
# prompt = ChatPromptTemplate.from_template(template_str)

# rag_chain = (
#     {"context": vector_db.as_retriever(search_kwargs={'k': 5}) | format_docs, "question": RunnablePassthrough()}
#     | prompt | llm | StrOutputParser()
# )

# # =========================================================
# # 5. Data Handling
# # =========================================================
# RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# os.makedirs(RESULTS_DIR, exist_ok=True)

# def get_synchronized_dataset(sample_ratio=0.01):
#     driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
#     db = os.environ.get("NEO4J_DATABASE", "graphdb")
#     query = "MATCH (c:Chunk) WHERE c.hf_id IS NOT NULL RETURN c.hf_id AS id"
#     with driver.session(database=db) as session:
#         result = session.run(query)
#         graph_ids = {str(record["id"]) for record in result if record["id"]}
#     driver.close()
    
#     dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
#     filtered_data = [row for row in dataset if str(row["id"]) in graph_ids]
#     return random.sample(filtered_data, max(1, int(len(filtered_data) * sample_ratio)))

# def build_test_cases(dataset):
#     test_cases = []
#     retriever = vector_db.as_retriever(search_kwargs={'k': 5})

#     for row in tqdm(dataset, desc="Generating Test Cases"):
#         try:
#             docs = retriever.invoke(row["question"])
#             retrieved_texts = [doc.page_content for doc in docs]
#             answer = rag_chain.invoke(row["question"])
            
#             ground_truth_context = row["documents"]
#             if isinstance(ground_truth_context, str):
#                 ground_truth_context = [ground_truth_context]
            
#             test_cases.append(LLMTestCase(
#                 input=row["question"],
#                 expected_output=row["response"],
#                 actual_output=answer,
#                 retrieval_context=retrieved_texts,
#                 context=ground_truth_context
#             ))
#         except Exception as e:
#             print(f"\n[SKIP] ID {row['id']} failed: {e}")
#             continue
#     return test_cases

# # =========================================================
# # 6. Main Execution & Dynamic Summary
# # =========================================================
# class GPTOSSJudge(DeepEvalBaseLLM):
#     def __init__(self, model): self.model = model
#     def load_model(self): return self.model
#     def get_model_name(self): return "gpt-oss"
#     def generate(self, prompt: str) -> str: return self.model.invoke(prompt).content
#     async def a_generate(self, prompt: str) -> str: return self.generate(prompt)

# if __name__ == "__main__":
#     test_data = get_synchronized_dataset(sample_ratio=0.01) 
#     test_cases = build_test_cases(test_data)
    
#     if test_cases:
#         judge = GPTOSSJudge(llm)
#         metrics = [ContextualRecallMetric(model=judge, threshold=0.7)]
#         active_metric_names = ", ".join([m.__class__.__name__.replace("Metric", "") for m in metrics])
        
#         # 1. Run Evaluation
#         results_container = deepeval.evaluate(
#             test_cases=test_cases, 
#             metrics=metrics,
#             hyperparameters={
#                 "model": "gpt-oss",
#                 "temperature": 0,
#                 "retrieval_k": 5,
#                 "index_name": "chunk_vector_index",
#                 "prompt_template": template_str
#             }
#         )

#         # 2. Access the internal list of results
#         # This fixes the 'EvaluationResult has no len()' error
#         test_results_list = results_container.test_results
        
#         final_results_file = os.path.join(RESULTS_DIR, "final_results.json")
#         ordered_results = []
#         metric_stats = {}
#         total_tests = len(test_results_list)
#         total_passed_cases = 0

#         # 3. Process each result
#         for result in test_results_list:
#             case_passed = result.success
#             if case_passed:
#                 total_passed_cases += 1

#             for m_data in result.metrics_data:
#                 m_name = m_data.name
#                 if m_name not in metric_stats:
#                     metric_stats[m_name] = {"passed": 0, "total": 0}
                
#                 metric_stats[m_name]["total"] += 1
#                 if m_data.success:
#                     metric_stats[m_name]["passed"] += 1

#             ordered_results.append({
#                 "status": "PASS" if case_passed else "FAIL",
#                 "input": result.input,
#                 "actual_output": result.actual_output,
#                 "expected_output": result.expected_output,
#                 "retrieval_context": result.retrieval_context,
#                 "ground_truth_context": result.context,
#                 "success": case_passed
#             })

#         # 4. Save Detailed Results
#         with open(final_results_file, "w", encoding="utf-8") as f:
#             json.dump(ordered_results, f, indent=2, ensure_ascii=False, default=str)

#         # 5. Save Summary
#         summary_file = os.path.join(RESULTS_DIR, "evaluation_summary.json")
#         summary_metrics = {
#             "overall_pass_rate": f"{(total_passed_cases / total_tests) * 100:.2f}%" if total_tests > 0 else "0.00%",
#             "total_test_cases": total_tests,
#             "passed_test_cases": total_passed_cases,
#             "failed_test_cases": total_tests - total_passed_cases,
#         }

#         for name, stats in metric_stats.items():
#             rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
#             summary_metrics[f"{name}_pass_rate"] = f"{rate:.2f}%"

#         summary_data = {
#             "summary_banner": f"Overall Metric Pass Rates - {active_metric_names}",
#             "overall_metrics": summary_metrics,
#             "detailed_results_path": final_results_file
#         }

#         with open(summary_file, "w", encoding="utf-8") as f:
#             json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
#         print(f"\n[INFO] Summary saved to: {summary_file}")
#     else:
#         print("[ERROR] No cases generated.")




###########################################################################################
###### This version is working well with small test cases and outputs both JSON files######
######                       With Full evaluation Crashed                            ######
###########################################################################################


# import os
# import json
# import random
# from typing import List
# from tqdm import tqdm
# from dotenv import load_dotenv
# from datasets import load_dataset
# from neo4j import GraphDatabase

# import deepeval
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import (
#     ContextualPrecisionMetric, 
#     ContextualRecallMetric, 
#     ContextualRelevancyMetric, 
#     AnswerRelevancyMetric, 
#     FaithfulnessMetric
# )
# from deepeval.models.base_model import DeepEvalBaseLLM

# from langchain_neo4j import Neo4jVector
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate

# # 1. Load configuration
# load_dotenv()

# # =========================================================
# # 2. Component Initialization
# # =========================================================
# class GatewayCompatibleEmbeddings(OpenAIEmbeddings):
#     def _get_len_safe_embeddings(self, texts: List[str], *, engine: str = None, **kwargs) -> List[List[float]]:
#         kwargs.pop("encoding_format", None)
#         kwargs.pop("chunk_size", None)
#         responses = self.client.create(input=texts, model=self.model, encoding_format=None, **kwargs)
#         return [d.embedding for d in responses.data]

# embeddings = GatewayCompatibleEmbeddings(
#     api_key=os.environ["EMBEDDING_MODEL_API_KEY"],
#     base_url=os.environ["EMBEDDING_MODEL_API_BASE"],
#     model="bge-m3"
# )

# llm = ChatOpenAI(
#     api_key=os.environ["UOS_API_KEY"],
#     base_url=os.environ["UOS_API_BASE"],
#     model="gpt-oss",
#     temperature=0
# )

# # =========================================================
# # 3. Chunk-First Graph Traversal Logic
# # =========================================================
# retrieval_query = """
# OPTIONAL MATCH (node)-[:HAS_ENTITY|MENTIONS|RELATED]->(e)
# OPTIONAL MATCH (e)-[r]->(neighbor)
# WITH node, score, 
#      collect(DISTINCT coalesce(e.name, e.id, labels(e)[0])) as entities, 
#      collect(DISTINCT type(r) + ' -> ' + coalesce(neighbor.name, neighbor.id, labels(neighbor)[0])) as relations
# RETURN 
#     node.text + "\\n\\nGRAPH ENRICHMENT:\\nEntities: " + apoc.text.join(entities, ", ") + 
#     "\\nRelationships: " + apoc.text.join(relations, ", ") AS text,
#     score, 
#     {source: node.source, hf_id: node.hf_id} AS metadata
# """

# vector_db = Neo4jVector.from_existing_index(
#     embedding=embeddings,
#     url=os.environ["NEO4J_URI"],
#     username=os.environ["NEO4J_USERNAME"],
#     password=os.environ["NEO4J_PASSWORD"],
#     database=os.environ.get("NEO4J_DATABASE", "graphdb"), 
#     index_name="chunk_vector_index", 
#     search_type="vector",
#     retrieval_query=retrieval_query
# )

# # =========================================================
# # 4. RAG Pipeline Construction
# # =========================================================
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# template = """Answer the question based only on the provided context:
# {context}

# Question: {question}
# Answer:"""

# prompt = ChatPromptTemplate.from_template(template)

# rag_chain = (
#     {"context": vector_db.as_retriever(search_kwargs={'k': 5}) | format_docs, "question": RunnablePassthrough()}
#     | prompt | llm | StrOutputParser()
# )

# # =========================================================
# # 5. Data Handling
# # =========================================================
# RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# os.makedirs(RESULTS_DIR, exist_ok=True)

# def get_synchronized_dataset(sample_ratio=1.0):
#     driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
#     db = os.environ.get("NEO4J_DATABASE", "graphdb")
#     query = "MATCH (c:Chunk) WHERE c.hf_id IS NOT NULL RETURN c.hf_id AS id"
#     with driver.session(database=db) as session:
#         result = session.run(query)
#         graph_ids = {str(record["id"]) for record in result if record["id"]}
#     driver.close()
    
#     dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
#     filtered_data = [row for row in dataset if str(row["id"]) in graph_ids]
#     return random.sample(filtered_data, max(1, int(len(filtered_data) * sample_ratio)))

# def build_test_cases(dataset):
#     test_cases = []
#     retriever = vector_db.as_retriever(search_kwargs={'k': 5})

#     for row in tqdm(dataset, desc="Generating Test Cases"):
#         try:
#             docs = retriever.invoke(row["question"])
#             retrieved_texts = [doc.page_content for doc in docs]
#             answer = rag_chain.invoke(row["question"])
            
#             ground_truth_context = row["documents"]
#             if isinstance(ground_truth_context, str):
#                 ground_truth_context = [ground_truth_context]
            
#             test_cases.append(LLMTestCase(
#                 input=row["question"],
#                 expected_output=row["response"],
#                 actual_output=answer,
#                 retrieval_context=retrieved_texts,
#                 context=ground_truth_context
#             ))
#         except Exception as e:
#             print(f"\n[SKIP] ID {row['id']} failed: {e}")
#             continue
#     return test_cases

# # =========================================================
# # 6. Main Execution & Dynamic Summary
# # =========================================================
# class GPTOSSJudge(DeepEvalBaseLLM):
#     def __init__(self, model): self.model = model
#     def load_model(self): return self.model
#     def get_model_name(self): return "gpt-oss"
#     def generate(self, prompt: str) -> str: return self.model.invoke(prompt).content
#     async def a_generate(self, prompt: str) -> str: return self.generate(prompt)

# if __name__ == "__main__":
#     test_data = get_synchronized_dataset(sample_ratio=1.0) 
#     test_cases = build_test_cases(test_data)
    
#     if test_cases:
#         judge = GPTOSSJudge(llm)
        
#         # ACTIVE METRICS DEFINED HERE
#         metrics = [
#             ContextualRecallMetric(model=judge, threshold=0.7),
#         ]
        
#         # Capture the metric names for the summary banner
#         active_metric_names = ", ".join([m.__class__.__name__.replace("Metric", "") for m in metrics])
        
#         # 1. Run Evaluation
#         deepeval.evaluate(test_cases=test_cases, metrics=metrics)

#         # 2. Prepare Detailed Results and Counters
#         final_results_file = os.path.join(RESULTS_DIR, "final_results.json")
#         ordered_results = []
#         metric_stats = {}
#         total_tests = len(test_cases)
#         total_passed_cases = 0

#         for tc in test_cases:
#             orig = tc.__dict__
#             m_results = {}
#             case_passed = True
            
#             if hasattr(tc, 'metrics_data') and tc.metrics_data:
#                 for m_data in tc.metrics_data:
#                     m_name = m_data.name
#                     if m_name not in metric_stats:
#                         metric_stats[m_name] = {"passed": 0, "total": 0}
                    
#                     metric_stats[m_name]["total"] += 1
#                     if m_data.success:
#                         metric_stats[m_name]["passed"] += 1
#                     else:
#                         case_passed = False

#                     m_results[m_name] = {
#                         "score": m_data.score,
#                         "reason": m_data.reason,
#                         "success": m_data.success
#                     }

#             if case_passed:
#                 total_passed_cases += 1

#             new_entry = {
#                 "status": "PASS" if case_passed else "FAIL",
#                 "input": orig.get("input"),
#                 "actual_output": orig.get("actual_output"),
#                 "expected_output": orig.get("expected_output"),
#                 "retrieval_context": orig.get("retrieval_context"),
#                 "ground truth context": orig.get("context"),
#                 # "metrics": m_results,
#                 # "token_cost": orig.get("token_cost"),
#                 # "completion_time": orig.get("completion_time")
#             }
#             ordered_results.append(new_entry)

#         # 3. Save Detailed Results
#         with open(final_results_file, "w", encoding="utf-8") as f:
#             json.dump(ordered_results, f, indent=2, ensure_ascii=False, default=str)

#         # 4. Save Summary with DYNAMIC Banner
#         summary_file = os.path.join(RESULTS_DIR, "evaluation_summary.json")
        
#         summary_metrics = {
#             "overall_pass_rate": f"{(total_passed_cases / total_tests) * 100:.2f}%" if total_tests > 0 else "0.00%",
#             "total_test_cases": total_tests,
#             "passed_test_cases": total_passed_cases,
#             "failed_test_cases": total_tests - total_passed_cases,
#         }

#         for name, stats in metric_stats.items():
#             rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
#             summary_metrics[f"{name}_pass_rate"] = f"{rate:.2f}%"

#         summary_data = {
#             "summary_banner": f"Overall Metric Pass Rates - {active_metric_names}",
#             "overall_metrics": summary_metrics,
#             "detailed_results_path": final_results_file
#         }

#         with open(summary_file, "w", encoding="utf-8") as f:
#             json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
#         print(f"\n[INFO] Summary saved with banner: {summary_data['summary_banner']}")
#     else:
#         print("[ERROR] No cases generated.")
































































#########################
###### OLD VERSION ######
#########################

# import os
# import json
# import random
# from typing import List
# from tqdm import tqdm
# from dotenv import load_dotenv
# from datasets import load_dataset
# from neo4j import GraphDatabase

# import deepeval
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import ContextualPrecisionMetric, FaithfulnessMetric
# from deepeval.models.base_model import DeepEvalBaseLLM

# from langchain_neo4j import Neo4jVector
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate

# # 1. Load configuration
# load_dotenv()

# # =========================================================
# # 2. Component Initialization
# # =========================================================
# class GatewayCompatibleEmbeddings(OpenAIEmbeddings):
#     def _get_len_safe_embeddings(self, texts: List[str], *, engine: str = None, **kwargs) -> List[List[float]]:
#         kwargs.pop("encoding_format", None)
#         kwargs.pop("chunk_size", None)
#         responses = self.client.create(input=texts, model=self.model, encoding_format=None, **kwargs)
#         return [d.embedding for d in responses.data]

# embeddings = GatewayCompatibleEmbeddings(
#     api_key=os.environ["EMBEDDING_MODEL_API_KEY"],
#     base_url=os.environ["EMBEDDING_MODEL_API_BASE"],
#     model="bge-m3"
# )

# llm = ChatOpenAI(
#     api_key=os.environ["UOS_API_KEY"],
#     base_url=os.environ["UOS_API_BASE"],
#     model="gpt-oss",
#     temperature=0
# )

# # =========================================================
# # 3. Chunk-First Graph Traversal Logic
# # =========================================================
# retrieval_query = """
# OPTIONAL MATCH (node)-[:HAS_ENTITY|MENTIONS|RELATED]->(e)
# OPTIONAL MATCH (e)-[r]->(neighbor)
# WITH node, score, 
#      collect(DISTINCT coalesce(e.name, e.id, labels(e)[0])) as entities, 
#      collect(DISTINCT type(r) + ' -> ' + coalesce(neighbor.name, neighbor.id, labels(neighbor)[0])) as relations
# RETURN 
#     node.text + "\\n\\nGRAPH ENRICHMENT:\\nEntities: " + apoc.text.join(entities, ", ") + 
#     "\\nRelationships: " + apoc.text.join(relations, ", ") AS text,
#     score, 
#     {source: node.source, hf_id: node.hf_id} AS metadata
# """

# vector_db = Neo4jVector.from_existing_index(
#     embedding=embeddings,
#     url=os.environ["NEO4J_URI"],
#     username=os.environ["NEO4J_USERNAME"],
#     password=os.environ["NEO4J_PASSWORD"],
#     database=os.environ.get("NEO4J_DATABASE", "graphdb"), 
#     index_name="chunk_vector_index", 
#     search_type="vector",
#     retrieval_query=retrieval_query
# )

# # =========================================================
# # 4. RAG Pipeline Construction
# # =========================================================
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# template = """Answer the question based only on the provided context:
# {context}

# Question: {question}
# Answer:"""

# prompt = ChatPromptTemplate.from_template(template)

# rag_chain = (
#     {"context": vector_db.as_retriever(search_kwargs={'k': 5}) | format_docs, "question": RunnablePassthrough()}
#     | prompt | llm | StrOutputParser()
# )

# # =========================================================
# # 5. Data Handling
# # =========================================================
# RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# os.makedirs(RESULTS_DIR, exist_ok=True)

# def get_synchronized_dataset(sample_ratio=1.0):
#     driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
#     db = os.environ.get("NEO4J_DATABASE", "graphdb")
#     query = "MATCH (c:Chunk) WHERE c.hf_id IS NOT NULL RETURN c.hf_id AS id"
#     with driver.session(database=db) as session:
#         result = session.run(query)
#         graph_ids = {str(record["id"]) for record in result if record["id"]}
#     driver.close()
    
#     dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
#     filtered_data = [row for row in dataset if str(row["id"]) in graph_ids]
#     return random.sample(filtered_data, max(1, int(len(filtered_data) * sample_ratio)))

# def build_test_cases(dataset):
#     test_cases = []
#     retriever = vector_db.as_retriever(search_kwargs={'k': 5})

#     for row in tqdm(dataset, desc="Generating Test Cases"):
#         try:
#             docs = retriever.invoke(row["question"])
#             retrieved_texts = [doc.page_content for doc in docs]
#             answer = rag_chain.invoke(row["question"])
            
#             ground_truth_context = row["documents"]
#             if isinstance(ground_truth_context, str):
#                 ground_truth_context = [ground_truth_context]
            
#             test_cases.append(LLMTestCase(
#                 input=row["question"],
#                 expected_output=row["response"],
#                 actual_output=answer,
#                 retrieval_context=retrieved_texts,
#                 context=ground_truth_context
#             ))
#         except Exception as e:
#             print(f"\n[SKIP] ID {row['id']} failed: {e}")
#             continue
#     return test_cases

# # =========================================================
# # 6. Execution and Export with Key Reordering
# # =========================================================
# class GPTOSSJudge(DeepEvalBaseLLM):
#     def __init__(self, model): self.model = model
#     def load_model(self): return self.model
#     def get_model_name(self): return "gpt-oss"
#     def generate(self, prompt: str) -> str: return self.model.invoke(prompt).content
#     async def a_generate(self, prompt: str) -> str: return self.generate(prompt)

# if __name__ == "__main__":
#     test_data = get_synchronized_dataset(sample_ratio=1.0) 
#     test_cases = build_test_cases(test_data)
    
#     if test_cases:
#         judge = GPTOSSJudge(llm)
#         metrics = [
#             # AnswerRelevancyMetric(model=judge, threshold=0.7),
#             # FaithfulnessMetric(model=judge, threshold=0.7),
#             ContextualPrecisionMetric(model=judge, threshold=0.7),
#             # ContextualRecallMetric(model=judge, threshold=0.7),
#             # ContextualRelevancyMetric(model=judge, threshold=0.7),
#         ]
        
#         # Run Evaluation
#         deepeval.evaluate(test_cases=test_cases, metrics=metrics)

#         # RENAME AND REORDER KEYS FOR FINAL JSON
#         final_results_file = os.path.join(RESULTS_DIR, "final_results.json")
        
#         ordered_results = []
#         for tc in test_cases:
#             orig = tc.__dict__
            
#             # We construct a new dict in the customized order
#             new_entry = {
#                 "input": orig.get("input"),
#                 "actual_output": orig.get("actual_output"),
#                 "expected_output": orig.get("expected_output"),
#                 "retrieval_context": orig.get("retrieval_context"),
#                 "ground truth context": orig.get("context"), # Renamed 
#                 "metadata": orig.get("metadata"),
#                 "tools_called": orig.get("tools_called"),
#                 "comments": orig.get("comments"),
#                 "expected_tools": orig.get("expected_tools"),
#                 "token_cost": orig.get("token_cost"),
#                 "completion_time": orig.get("completion_time"),
#                 "multimodal": orig.get("multimodal", False),
#                 "name": orig.get("name"),
#                 "tags": orig.get("tags"),
#                 "mcp_servers": orig.get("mcp_servers"),
#                 "mcp_tools_called": orig.get("mcp_tools_called"),
#                 "mcp_resources_called": orig.get("mcp_resources_called"),
#                 "mcp_prompts_called": orig.get("mcp_prompts_called"),
#                 "custom_column_key_values": orig.get("custom_column_key_values")
#             }
#             ordered_results.append(new_entry)

#         with open(final_results_file, "w", encoding="utf-8") as f:
#             json.dump(ordered_results, f, indent=2, ensure_ascii=False, default=str)
            
#         print(f"\n[INFO] Results saved to {final_results_file}")
#     else:
#         print("[ERROR] No cases generated.")






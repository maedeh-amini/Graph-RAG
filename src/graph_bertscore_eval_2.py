

#This version worked for gpt-oss

# src/graph_bertscore_eval_2.py
import os
import json
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
import torch

from bert_score import score, BERTScorer
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from src.graph_rag_query_2 import GraphRAGQuery

# =========================================================
# Load environment variables
# =========================================================
load_dotenv()
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
UOS_API_KEY = os.environ["UOS_API_KEY"]
UOS_API_BASE = os.environ["UOS_API_BASE"]

# =========================================================
# Helper: get existing graph (read-only)
# =========================================================
def get_graph():
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
        sanitize=True
    )

# =========================================================
# Load 5% of HotpotQA test set
# =========================================================
def load_test_dataset(sample_ratio=0.05):
    dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["question", "response"]]
    )
    dataset = dataset.shuffle(seed=42)
    sample_size = max(1, int(len(dataset) * sample_ratio))
    dataset = dataset.select(range(sample_size))
    test_examples = [
        {"question": row["question"], "reference": row["response"]}
        for row in dataset
    ]
    print(f"[INFO] Loaded {len(test_examples)} test examples ({sample_ratio*100}% of test set)")
    return test_examples

# =========================================================
# Evaluate Graph-RAG with BERTScore
# =========================================================
def evaluate_graph_rag_with_bertscore(graph_rag, test_examples):
    references = []
    predictions = []

    for example in tqdm(test_examples, desc="Evaluating Graph-RAG on test set"):
        question = example["question"]
        reference = example["reference"]

        try:
            rag_output = graph_rag.query(question)
            if isinstance(rag_output, dict):
                predicted_answer = rag_output.get("result", "")
            else:
                predicted_answer = str(rag_output)
        except Exception as e:
            print("[ERROR] Query failed:", e)
            predicted_answer = ""

        predictions.append(predicted_answer)
        references.append(reference)

    # Use BERTScorer object for compatibility with transformers v5+
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = BERTScorer(
        model_type="distilroberta-base",
        lang="en",
        rescale_with_baseline=False,
        device=device
    )
    P, R, F1 = scorer.score(predictions, references)

    avg_precision = P.mean().item()
    avg_recall = R.mean().item()
    avg_f1 = F1.mean().item()

    print(
        f"\n[RESULT] BERTScore - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}"
    )
    return avg_precision, avg_recall, avg_f1

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    # Graph object (read-only)
    graph = get_graph()

    # Initialize LLMs directly
    qa_llm = ChatOpenAI(
        api_key=UOS_API_KEY,
        base_url=UOS_API_BASE,
        model="gemma3",
        temperature=0.2
    )
    cypher_llm = ChatOpenAI(
        api_key=UOS_API_KEY,
        base_url=UOS_API_BASE,
        model="gemma3",
        temperature=0.2
    )

    # Graph-RAG query object (does NOT modify DB)
    graph_rag = GraphRAGQuery(graph=graph, qa_llm=qa_llm, cypher_llm=cypher_llm)

    # Load 5% of test set
    test_data = load_test_dataset(sample_ratio=0.05)

    # Evaluate
    evaluate_graph_rag_with_bertscore(graph_rag, test_data)






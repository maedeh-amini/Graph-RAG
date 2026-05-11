import os
import torch
import random
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from bert_score import BERTScorer
from neo4j import GraphDatabase

# Import your refined query components
from src.graph_query_2 import rag_chain

# =========================================================
# 1. Configuration & Setup
# =========================================================
load_dotenv()
DB_NAME = os.environ.get("NEO4J_DATABASE", "graphdb")

# --- TUNE SAMPLE SIZE ---
SAMPLE_RATIO =  1.0
# ----------------------------------

def get_ids_from_graph():
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    query = "MATCH (c:Chunk) WHERE c.hf_id IS NOT NULL RETURN c.hf_id AS id"
    
    with driver.session(database=DB_NAME) as session:
        result = session.run(query)
        # Using a set to ensure we don't duplicate IDs if a chunk was split
        ids = list(set([record["id"] for record in result]))
    
    driver.close()
    print(f"[INFO] Found {len(ids)} unique IDs in the '{DB_NAME}' database.")
    return ids

# =========================================================
# 2. Dataset Filtering
# =========================================================
def load_synchronized_test_set():
    graph_ids = get_ids_from_graph()
    if not graph_ids:
        raise ValueError("Graph is empty! Run the builder script first.")

    dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
    
    # Filter dataset to only include rows present in our Neo4j DB
    filtered_data = dataset.filter(lambda row: str(row["id"]) in graph_ids)
    
    test_examples = [
        {"question": row["question"], "reference": row["response"], "id": row["id"]}
        for row in filtered_data
    ]
    return test_examples

# =========================================================
# 3. BERTScore Evaluation Logic
# =========================================================
def run_evaluation(test_examples, label="Evaluation"):
    if not test_examples:
        print(f"[SKIP] No examples for {label}")
        return

    predictions = []
    references = []

    print(f"\n[START] {label} | Samples: {len(test_examples)}")
    
    for example in tqdm(test_examples, desc=f"Processing {label}"):
        try:
            ans = rag_chain.invoke(example["question"])
            predictions.append(ans)
            references.append(example["reference"])
        except Exception as e:
            print(f"\n[ERROR] Row {example['id']} failed: {e}")
            predictions.append("") 
            references.append(example["reference"])

    # BERTScore Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = BERTScorer(model_type="distilroberta-base", lang="en", device=device)
    
    P, R, F1 = scorer.score(predictions, references)

    avg_p, avg_r, avg_f1 = P.mean().item(), R.mean().item(), F1.mean().item()

    print("\n" + "="*45)
    print(f"FINAL STATS: {label}")
    print("="*45)
    print(f"Precision: {avg_p:.4f}")
    print(f"Recall:    {avg_r:.4f}")
    print(f"F1 Score:  {avg_f1:.4f}")
    print("="*45 + "\n")

# =========================================================
# Main Execution
# =========================================================
if __name__ == "__main__":
    try:
        # 1. Load everything available in the graph
        full_data = load_synchronized_test_set()
        total_available = len(full_data)

        # 2. Determine sample size based on your tuning variable
        sample_size = max(1, int(total_available * SAMPLE_RATIO))
        
        # If SAMPLE_RATIO is 1.0, we just run the whole thing once
        if SAMPLE_RATIO >= 1.0:
            run_evaluation(full_data, label="FULL GRAPH EVALUATION (100%)")
        else:
            # Otherwise, run the small sample first as a "sneak peek"
            sampled_data = random.sample(full_data, sample_size)
            print(f"[TUNING] Current Sample Ratio: {SAMPLE_RATIO*100}%")
            
            run_evaluation(sampled_data, label=f"SAMPLED EVALUATION ({SAMPLE_RATIO*100}%)")
            
            # Optional: Ask user if they want to proceed to full data
            proceed = input(f"Proceed to Full Evaluation of {total_available} rows? (y/n): ")
            if proceed.lower() == 'y':
                run_evaluation(full_data, label="FULL GRAPH EVALUATION (100%)")

    except Exception as e:
        print(f"[CRITICAL FAILURE]: {e}")





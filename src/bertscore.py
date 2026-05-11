
import json
from datasets import load_dataset
from tqdm import tqdm
from bert_score import score
from src.search import RAGSearch

def load_test_dataset():
    """
    Load the Hugging Face dataset and extract the 20% test split.
    Returns a list of dictionaries: {"question": ..., "reference": ...}
    """
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

    # Concatenate splits if necessary
    if isinstance(dataset, dict):
        dataset = dataset["train"]  # train_test_split will be applied manually

    # Keep only instruction and response
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["instruction", "response"]])

    # Perform train/test split
    split = dataset.train_test_split(test_size=0.01, shuffle=True)  #test-split= 0.8% 
    test_dataset = split["test"]

    # Convert to list of dicts
    test_examples = [{"question": row["instruction"], "reference": row["response"]} for row in test_dataset]
    print(f"[INFO] Loaded {len(test_examples)} test examples")
    return test_examples


def evaluate_rag_with_bertscore(rag_search: RAGSearch, test_examples):
    """
    Runs RAG on test examples and computes BERTScore.
    Returns average precision, recall, f1.
    """
    references = []
    predictions = []

    for example in tqdm(test_examples, desc="Evaluating RAG on test set"):
        question = example["question"]
        reference = example["reference"]

        # RAG query
        rag_output = rag_search.query_rag(question, top_k=5)
        rag_output_json = json.loads(rag_output)
        predicted_answer = rag_output_json.get("answer", "")

        predictions.append(predicted_answer)
        references.append(reference)

    # DEBUG PRINT: show question, prediction, reference 
    #     print("Q:", question)
    #     print("Pred:", predicted_answer)
    #     print("Ref :", reference)
    #     print("-" * 50)

    #Compute BERTScore
    P, R, F1 = score(
        predictions,
        references,
        lang="en",
        model_type="distilroberta-base",
        batch_size=32, 
        rescale_with_baseline=True
    )
    avg_precision = P.mean().item()
    avg_recall = R.mean().item()
    avg_f1 = F1.mean().item()

    print(f"\n[RESULT] BERTScore - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
    return avg_precision, avg_recall, avg_f1


if __name__ == "__main__":
    # Initialize RAG pipeline
    rag_search = RAGSearch()

    # Load test data
    test_data = load_test_dataset()

    # Evaluate
    evaluate_rag_with_bertscore(rag_search, test_data)

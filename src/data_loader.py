from pathlib import Path
from typing import List, Any
from langchain_core.documents import Document
from datasets import load_dataset


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported local files from the data directory as well as Hugging Face dataset and convert them to LangChain document structure.
    Local File Support: PDF, TXT, CSV, Excel, Word, JSON
    HF Dataset support: "hotpotqa/hotpot_qa", "fullwiki"
    """

    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")

    documents = []

    # ------------------------
    # 1. Load Hugging Face dataset
    # ------------------------

    try:
        # Load dataset
        dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki")

        # Check split name
        split_name = list(dataset.keys())[0]  # usually 'train'
        print(f"Using split: {split_name}")

        # Take 0.1% of the data
        subset = dataset[split_name].select(range(int(0.0001 * len(dataset[split_name]))))
        subset = subset.filter(lambda row: row["type"] == "bridge")  # only bridge questions


        def extract_title_passages(context_obj, min_tokens=20):
            """
            context_obj schema:
            {
            "title": [t1, t2, ...],
            "sentences": [[s1], [s2], ...]
            }
            Returns list of (title, passage_text)
            """
            titles = context_obj["title"]
            sentence_groups = context_obj["sentences"]

            assert len(titles) == len(sentence_groups), "title-sentence alignment broken"

            passages = []
            for title, sentences in zip(titles, sentence_groups):
                text = " ".join(s.strip() for s in sentences if s and s.strip())
                if len(text.split()) >= min_tokens:
                    passages.append((title, text))
            return passages


        # Build Documents from context (NOT QA pairs)
        documents = []
        doc_id = 0

        for row in subset:
            title_passages = extract_title_passages(row["context"])

            for title, passage_text in title_passages:
                documents.append(
                    Document(
                        page_content=passage_text,
                        metadata={
                            "doc_id": doc_id,
                            "example_id": row["id"],
                            "title": title,
                        }
                    )
                )
                doc_id += 1

        print(f"Total passages indexed: {len(documents)}")
        print(documents[:2])

    except Exception as e:
        print(f"[ERROR] Failed to load Hugging Face dataset: {e}")

    return documents

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)
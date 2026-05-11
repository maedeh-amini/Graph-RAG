import os
import random
import asyncio
from tqdm import tqdm
from datasets import load_dataset
from neo4j import GraphDatabase
from dotenv import load_dotenv

from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings 
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline, OnError
from neo4j_graphrag.indexes import create_vector_index

# =========================================================
# 1. Setup & Environment
# =========================================================
load_dotenv()

UOS_API_KEY = os.environ["UOS_API_KEY"]
UOS_API_BASE = os.environ["UOS_API_BASE"]
EMB_KEY = os.environ["EMBEDDING_MODEL_API_KEY"]
EMB_BASE = os.environ["EMBEDDING_MODEL_API_BASE"]
DB_NAME = os.environ.get("NEO4J_DATABASE", "graphdb")

class GatewayCompatibleEmbeddings(OpenAIEmbeddings):
    def embed_query(self, text):
        return self.client.embeddings.create(
            input=text, 
            model=self.model,
            encoding_format=None
        ).data[0].embedding

    def embed_nodes(self, texts):
        response = self.client.embeddings.create(
            input=texts, 
            model=self.model,
            encoding_format=None
        )
        return [d.embedding for d in response.data]

neo4j_driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

# =========================================================
# 2. Knowledge Graph Orchestrator
# =========================================================
class KnowledgeGraphOrchestrator:
    def __init__(self, driver, subset_fraction=1.0):
        self.driver = driver
        self.subset_fraction = subset_fraction
        self.processed_data = [] # List of dicts: {'id': ..., 'text': ...}

        self.llm = OpenAILLM(
            model_name="gpt-oss", 
            api_key=UOS_API_KEY,
            base_url=UOS_API_BASE,
            model_params={"temperature": 0}
        )

        self.embedder = GatewayCompatibleEmbeddings(model="bge-m3")
        self.embedder.client.api_key = EMB_KEY
        self.embedder.client.base_url = EMB_BASE

    def load_source_data(self):
        dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
        subset_size = max(1, int(len(dataset) * self.subset_fraction))
        indices = random.sample(range(len(dataset)), subset_size)
        
        for idx in tqdm(indices, desc="Fetching HF Data"):
            row = dataset[idx]
            
            # CAPTURE THE ID HERE
            hf_id = str(row["id"]) 
            
            # Capture and clean the documents text
            docs = row["documents"]
            content = " ".join(docs) if isinstance(docs, list) else str(docs)
            
            self.processed_data.append({
                "hf_id": hf_id,
                "text": content
            })
        print(f"[INFO] Loaded {len(self.processed_data)} items with original IDs.")

    async def execute_pipeline(self):
        # 1. Verification of Index
        try:
            create_vector_index(
                self.driver,
                name="entity_vector_index",
                label="__Entity__",
                embedding_property="embedding",
                dimensions=1024, 
                similarity_fn="cosine"
            )
        except Exception:
            pass

        # 2. Initialize Pipeline
        kg_pipeline = SimpleKGPipeline(
            llm=self.llm,
            driver=self.driver,
            neo4j_database=DB_NAME,
            embedder=self.embedder,
            from_pdf=False,
            schema="FREE", 
            on_error=OnError.IGNORE
        )

        print(f"[INFO] Building Graph & Injecting IDs into {DB_NAME}...")
        
        # We process one-by-one to maintain the ID mapping
        for item in tqdm(self.processed_data, desc="Processing Rows"):
            try:
                # A. Run Extraction
                await kg_pipeline.run_async(text=item['text'])
                
                # B. Immediate ID Injection
                # We find the node by text and set the hf_id property
                injection_query = """
                MATCH (c:Chunk) 
                WHERE c.text = $text AND c.hf_id IS NULL
                SET c.hf_id = $hf_id
                """
                
                with self.driver.session(database=DB_NAME) as session:
                    session.run(injection_query, text=item['text'], hf_id=item['hf_id'])
                
                # Gateway cooldown
                await asyncio.sleep(1.5) 
                
            except Exception as e:
                print(f"[ERROR] Row {item['hf_id']} failed: {e}")

# =========================================================
# 3. Execution
# =========================================================
if __name__ == "__main__":
    # IMPORTANT: Clear database before re-running to avoid ID mismatches
    # Command: MATCH (n) DETACH DELETE n
    
    orchestrator = KnowledgeGraphOrchestrator(neo4j_driver, subset_fraction=1.0
    )
    orchestrator.load_source_data()
    asyncio.run(orchestrator.execute_pipeline())
    print("[SUCCESS] Builder Pipeline Finished with ID Persistence.")














# import os
# import random
# import asyncio
# from tqdm import tqdm
# from datasets import load_dataset
# from neo4j import GraphDatabase
# from dotenv import load_dotenv

# from neo4j_graphrag.llm import OpenAILLM
# from neo4j_graphrag.embeddings import OpenAIEmbeddings 
# from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline, OnError
# from neo4j_graphrag.indexes import create_vector_index

# # =========================================================
# # 1. Setup & Environment
# # =========================================================
# load_dotenv()

# UOS_API_KEY = os.environ["UOS_API_KEY"]
# UOS_API_BASE = os.environ["UOS_API_BASE"]
# EMB_KEY = os.environ["EMBEDDING_MODEL_API_KEY"]
# EMB_BASE = os.environ["EMBEDDING_MODEL_API_BASE"]

# # Custom Embedder Class to bypass LiteLLM 'base64' error
# class LiteLLMEmbeddings(OpenAIEmbeddings):
#     def embed_query(self, text):
#         return self.client.embeddings.create(
#             input=text, 
#             model=self.model,
#             encoding_format=None
#         ).data[0].embedding

#     def embed_nodes(self, texts):
#         response = self.client.embeddings.create(
#             input=texts, 
#             model=self.model,
#             encoding_format=None
#         )
#         return [d.embedding for d in response.data]

# neo4j_driver = GraphDatabase.driver(
#     os.environ["NEO4J_URI"],
#     auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
# )

# class ModernGraphRAGBuilder:
#     def __init__(self, driver, subset_fraction=0.01):
#         self.driver = driver
#         self.subset_fraction = subset_fraction
#         self.raw_texts = []

#         # 1. LLM Setup
#         self.llm = OpenAILLM(
#             model_name="gpt-oss", 
#             api_key=UOS_API_KEY,
#             base_url=UOS_API_BASE,
#             model_params={"temperature": 0}
#         )

#         # 2. BGE-M3 Embedder Setup (Using Custom Class)
#         self.embedder = LiteLLMEmbeddings(model="bge-m3")
#         self.embedder.client.api_key = EMB_KEY
#         self.embedder.client.base_url = EMB_BASE

#     def load_from_huggingface(self):
#         dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
#         subset_size = max(1, int(len(dataset) * self.subset_fraction))
#         indices = random.sample(range(len(dataset)), subset_size)
        
#         for idx in tqdm(indices, desc="Fetching HF Data"):
#             row = dataset[idx]
#             chunk = row["documents"]
#             if isinstance(chunk, str): self.raw_texts.append(chunk)
#             elif isinstance(chunk, list): self.raw_texts.append(" ".join(chunk))
#         print(f"[INFO] Loaded {len(self.raw_texts)} chunks.")

#     async def build_and_embed(self):
#         try:
#             create_vector_index(
#                 self.driver,
#                 name="entity_vector_index",
#                 label="__Entity__",
#                 embedding_property="embedding",
#                 dimensions=1024, 
#                 similarity_fn="cosine"
#             )
#             print("[INFO] Vector index verified.")
#         except Exception:
#             print("[INFO] Vector index already exists.")

#         kg_pipeline = SimpleKGPipeline(
#             llm=self.llm,
#             driver=self.driver,
#             neo4j_database=os.getenv("NEO4J_DATABASE"),
#             embedder=self.embedder,
#             from_pdf=False,
#             schema="FREE", 
#             on_error=OnError.IGNORE
#         )

#         # Batching logic
#         batch_size = 2 
#         batches = [self.raw_texts[i : i + batch_size] for i in range(0, len(self.raw_texts), batch_size)]
        
#         print(f"[INFO] Starting KG Construction ({len(batches)} batches)...")
        
#         # New Progress Bar for Batches
#         pbar = tqdm(total=len(batches), desc="Building Knowledge Graph")
        
#         for batch in batches:
#             batch_text = "\n\n".join(batch)
#             try:
#                 await kg_pipeline.run_async(text=batch_text)
#                 await asyncio.sleep(2) # Gateway cooldown
#                 pbar.update(1) # Increment progress bar
#             except Exception as e:
#                 # We update the bar even on failure so the total count remains accurate
#                 pbar.set_postfix_str(f"Last Error: {str(e)[:20]}...")
#                 pbar.update(1)
                
#         pbar.close()

# if __name__ == "__main__":
#     builder = ModernGraphRAGBuilder(neo4j_driver, subset_fraction=0.01)
#     builder.load_from_huggingface()
#     asyncio.run(builder.build_and_embed())
#     print("[SUCCESS] Builder Finished.")




























# import os
# import random
# import asyncio
# from tqdm import tqdm
# from datasets import load_dataset
# from neo4j import GraphDatabase
# from dotenv import load_dotenv

# from neo4j_graphrag.llm import OpenAILLM
# from neo4j_graphrag.embeddings import OpenAIEmbeddings 
# from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline, OnError
# from neo4j_graphrag.indexes import create_vector_index

# # =========================================================
# # 1. Setup & Environment
# # =========================================================
# load_dotenv()

# UOS_API_KEY = os.environ["UOS_API_KEY"]
# UOS_API_BASE = os.environ["UOS_API_BASE"]
# EMB_KEY = os.environ["EMBEDDING_MODEL_API_KEY"]
# EMB_BASE = os.environ["EMBEDDING_MODEL_API_BASE"]

# # Custom Embedder Class to bypass LiteLLM 'base64' error
# class LiteLLMEmbeddings(OpenAIEmbeddings):
#     def embed_query(self, text):
#         # Force encoding_format to None in the raw API call
#         return self.client.embeddings.create(
#             input=text, 
#             model=self.model,
#             encoding_format=None
#         ).data[0].embedding

#     def embed_nodes(self, texts):
#         # Force encoding_format to None for batches
#         response = self.client.embeddings.create(
#             input=texts, 
#             model=self.model,
#             encoding_format=None
#         )
#         return [d.embedding for d in response.data]

# neo4j_driver = GraphDatabase.driver(
#     os.environ["NEO4J_URI"],
#     auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
# )

# class ModernGraphRAGBuilder:
#     def __init__(self, driver, subset_fraction=0.01):
#         self.driver = driver
#         self.subset_fraction = subset_fraction
#         self.raw_texts = []

#         # 1. LLM Setup
#         self.llm = OpenAILLM(
#             model_name="gpt-oss", 
#             api_key=UOS_API_KEY,
#             base_url=UOS_API_BASE,
#             model_params={"temperature": 0}
#         )

#         # 2. BGE-M3 Embedder Setup (Using our Custom Class)
#         self.embedder = LiteLLMEmbeddings(model="bge-m3")
#         self.embedder.client.api_key = EMB_KEY
#         self.embedder.client.base_url = EMB_BASE

#     def load_from_huggingface(self):
#         dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
#         subset_size = max(1, int(len(dataset) * self.subset_fraction))
#         indices = random.sample(range(len(dataset)), subset_size)
#         for idx in tqdm(indices, desc="Fetching HF Data"):
#             row = dataset[idx]
#             chunk = row["documents"]
#             if isinstance(chunk, str): self.raw_texts.append(chunk)
#             elif isinstance(chunk, list): self.raw_texts.append(" ".join(chunk))
#         print(f"[INFO] Loaded {len(self.raw_texts)} chunks.")

#     async def build_and_embed(self):
#         try:
#             create_vector_index(
#                 self.driver,
#                 name="entity_vector_index",
#                 label="__Entity__",
#                 embedding_property="embedding",
#                 dimensions=1024, 
#                 similarity_fn="cosine"
#             )
#             print("[INFO] Vector index verified.")
#         except Exception:
#             print("[INFO] Vector index already exists.")

#         kg_pipeline = SimpleKGPipeline(
#             llm=self.llm,
#             driver=self.driver,
#             neo4j_database=os.getenv("NEO4J_DATABASE"),
#             embedder=self.embedder,
#             from_pdf=False,
#             schema="FREE", 
#             on_error=OnError.IGNORE
#         )

#         batch_size = 2 
#         print("[INFO] Starting KG Construction...")
#         for i in range(0, len(self.raw_texts), batch_size):
#             batch_text = "\n\n".join(self.raw_texts[i : i + batch_size])
#             print(f"Processing Batch {i//batch_size + 1}...")
#             try:
#                 # We use the pipeline, which will now call our 'LiteLLMEmbeddings'
#                 await kg_pipeline.run_async(text=batch_text)
#                 await asyncio.sleep(2) 
#             except Exception as e:
#                 print(f"[WARNING] Batch failed: {e}")

# if __name__ == "__main__":
#     builder = ModernGraphRAGBuilder(neo4j_driver, subset_fraction=0.01)
#     builder.load_from_huggingface()
#     asyncio.run(builder.build_and_embed())
#     print("[SUCCESS] Builder Finished.")





















# import os
# import random
# import asyncio
# from tqdm import tqdm
# from datasets import load_dataset
# from neo4j import GraphDatabase
# from dotenv import load_dotenv

# from neo4j_graphrag.llm import OpenAILLM
# from neo4j_graphrag.embeddings import OpenAIEmbeddings # Changed to API-based
# from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline, OnError
# from neo4j_graphrag.indexes import create_vector_index

# # =========================================================
# # 1. Setup & Environment
# # =========================================================
# load_dotenv()

# UOS_API_KEY = os.environ["UOS_API_KEY"]
# UOS_API_BASE = os.environ["UOS_API_BASE"]
# EMBEDDING_MODEL_API_KEY=os.environ["EMBEDDING_MODEL_API_KEY"]
# EMBEDDING_MODEL_API_BASE=os.environ["EMBEDDING_MODEL_API_BASE"]


# neo4j_driver = GraphDatabase.driver(
#     os.environ["NEO4J_URI"],
#     auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
# )

# class ModernGraphRAGBuilder:
#     def __init__(self, driver, subset_fraction=0.01):
#         self.driver = driver
#         self.subset_fraction = subset_fraction
#         self.raw_texts = []

#         # 1. LLM (UOS Gateway)
#         self.llm = OpenAILLM(
#             model_name="gpt-oss", 
#             api_key=UOS_API_KEY,
#             base_url=UOS_API_BASE,
#             model_params={"temperature": 0}
#         )

#         # 2. BGE-M3 Embedder (UOS Gateway)
#         # BGE-M3 typically uses 1024 dimensions
#         self.embedder = OpenAIEmbeddings(
#             api_key=EMBEDDING_MODEL_API_KEY,
#             base_url=EMBEDDING_MODEL_API_BASE,
#             model="bge-m3" 
#         )

#     def load_from_huggingface(self):
#         dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
#         subset_size = max(1, int(len(dataset) * self.subset_fraction))
#         indices = random.sample(range(len(dataset)), subset_size)

#         for idx in tqdm(indices, desc="Fetching HF Data"):
#             row = dataset[idx]
#             chunk = row["documents"]
#             if isinstance(chunk, str): self.raw_texts.append(chunk)
#             elif isinstance(chunk, list): self.raw_texts.append(" ".join(chunk))
#         print(f"[INFO] Loaded {len(self.raw_texts)} chunks.")

#     async def build_and_embed(self):
#         # Create Vector Index for BGE-M3 (1024 dims)
#         try:
#             create_vector_index(
#                 self.driver,
#                 name="entity_vector_index",
#                 label="__Entity__",
#                 embedding_property="embedding",
#                 dimensions=1024, # Updated for BGE-M3
#                 similarity_fn="cosine"
#             )
#             print("[INFO] Vector index created (1024 dimensions).")
#         except Exception:
#             print("[INFO] Vector index exists (Ensure it is 1024 dims).")

#         kg_pipeline = SimpleKGPipeline(
#             llm=self.llm,
#             driver=self.driver,
#             neo4j_database=os.getenv("NEO4J_DATABASE"),
#             embedder=self.embedder,
#             from_pdf=False,
#             schema="FREE", 
#             on_error=OnError.IGNORE
#         )

#         # Batch processing to prevent University Gateway timeouts
#         batch_size = 2 
#         print("[INFO] Starting KG Construction...")
#         for i in range(0, len(self.raw_texts), batch_size):
#             batch_text = "\n\n".join(self.raw_texts[i : i + batch_size])
#             print(f"Processing Batch {i//batch_size + 1}...")
#             try:
#                 await kg_pipeline.run_async(text=batch_text)
#                 await asyncio.sleep(2) # Gateway cooldown
#             except Exception as e:
#                 print(f"[WARNING] Batch failed: {e}")

# if __name__ == "__main__":
#     builder = ModernGraphRAGBuilder(neo4j_driver, subset_fraction=0.01)
#     builder.load_from_huggingface()
#     asyncio.run(builder.build_and_embed())

#     if builder.raw_texts:
#         asyncio.run(builder.build_and_embed())
#         print("[SUCCESS] Builder Finished.")
#     else:
#         print("[WARN] No texts loaded. Check dataset or subset_fraction.")






# import os
# import random
# import asyncio
# from tqdm import tqdm
# from datasets import load_dataset
# from neo4j import GraphDatabase
# from dotenv import load_dotenv

# from neo4j_graphrag.llm import OpenAILLM
# from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
# from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline, OnError
# from neo4j_graphrag.indexes import create_vector_index

# load_dotenv()

# class ModernGraphRAGBuilder:
#     def __init__(self, subset_fraction=1.0):
#         self.driver = GraphDatabase.driver(
#             os.environ["NEO4J_URI"],
#             auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
#         )
#         self.subset_fraction = subset_fraction
#         self.raw_texts = []

#         # LLM configured for University Gateway
#         self.llm = OpenAILLM(
#             model_name="gpt-oss", 
#             api_key=os.environ["UOS_API_KEY"],
#             base_url=os.environ["UOS_API_BASE"],
#             model_params={"temperature": 0}
#         )

#         # Local Embedder (all-MiniLM-L6-v2)
#         self.embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")

#     def load_from_huggingface(self):
#         dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
#         subset_size = max(1, int(len(dataset) * self.subset_fraction))
#         indices = random.sample(range(len(dataset)), subset_size)

#         for idx in tqdm(indices, desc="Fetching HF Data"):
#             chunk = dataset[idx]["documents"]
#             if isinstance(chunk, str): 
#                 self.raw_texts.append(chunk)
#             elif isinstance(chunk, list): 
#                 self.raw_texts.append(" ".join(chunk))
        
#         print(f"[INFO] Loaded {len(self.raw_texts)} text chunks.")

#     async def build_and_embed(self):
#         # 1. Create Vector Index (384 dimensions)
#         try:
#             create_vector_index(
#                 self.driver,
#                 name="entity_vector_index",
#                 label="__Entity__",  # Pipeline default
#                 embedding_property="embedding",
#                 dimensions=384, 
#                 similarity_fn="cosine"
#             )
#             print("[INFO] Vector index created/verified.")
#         except Exception:
#             pass 

#         # 2. Setup Pipeline in FREE mode
#         kg_pipeline = SimpleKGPipeline(
#             llm=self.llm,
#             driver=self.driver,
#             neo4j_database=os.getenv("NEO4J_DATABASE"),
#             embedder=self.embedder,
#             from_pdf=False,
#             schema="FREE", 
#             on_error=OnError.IGNORE
#         )

#         # 3. Batch Processing with Progress Bar
#         batch_size = 2 
#         num_batches = (len(self.raw_texts) + batch_size - 1) // batch_size
        
#         print(f"[INFO] Starting KG Construction ({num_batches} batches)...")
        
#         # tqdm wrap for the batch loop
#         for i in tqdm(range(0, len(self.raw_texts), batch_size), total=num_batches, desc="Building Graph"):
#             batch_text = "\n\n".join(self.raw_texts[i : i + batch_size])
            
#             try:
#                 await kg_pipeline.run_async(text=batch_text)
#                 # Polite delay for the API gateway
#                 await asyncio.sleep(2) 
#             except Exception as e:
#                 # tqdm.write prevents the error message from breaking the progress bar UI
#                 tqdm.write(f"[ERROR] Skipping Batch {i//batch_size + 1}: {e}")

# if __name__ == "__main__":
#     # subset_fraction=0.1 runs 10% of the data for testing
#     builder = ModernGraphRAGBuilder(subset_fraction=0.1) 
#     builder.load_from_huggingface()
    
#     if builder.raw_texts:
#         asyncio.run(builder.build_and_embed())
#         print("[SUCCESS] Builder Finished.")
#     else:
#         print("[WARN] No texts loaded. Check dataset or subset_fraction.")








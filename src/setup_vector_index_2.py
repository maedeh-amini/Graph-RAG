
import os
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Configuration
uri = os.environ["NEO4J_URI"]
user = os.environ["NEO4J_USERNAME"]
password = os.environ["NEO4J_PASSWORD"]
db_name = os.environ.get("NEO4J_DATABASE", "test01") 

driver = GraphDatabase.driver(uri, auth=(user, password))

def setup_indexes():
    # 1. Index for extracted Graph Entities (used for structural reasoning)
    entity_index_query = """
    CREATE VECTOR INDEX entity_vector_index IF NOT EXISTS
    FOR (n:__Entity__)
    ON (n.embedding)
    OPTIONS {indexConfig: {
      `vector.dimensions`: 1024,
      `vector.similarity_function`: 'cosine'
    }}
    """
    
    # 2. Index for Raw Text Chunks (used for standard RAG retrieval)
    chunk_index_query = """
    CREATE VECTOR INDEX chunk_vector_index IF NOT EXISTS
    FOR (n:Chunk)
    ON (n.embedding)
    OPTIONS {indexConfig: {
      `vector.dimensions`: 1024,
      `vector.similarity_function`: 'cosine'
    }}
    """
    
    print(f"[INFO] Connecting to {uri} | Database: {db_name}")
    
    with driver.session(database=db_name) as session:
        try:
            # Create Entity Index
            print(f"Creating 'entity_vector_index' on {db_name}...")
            session.run(entity_index_query)
            
            # Create Chunk Index
            print(f"Creating 'chunk_vector_index' on {db_name}...")
            session.run(chunk_index_query)
            
            # Verification Loop
            print("Verifying index states...")
            for i in range(5):
                time.sleep(2)
                result = session.run("SHOW VECTOR INDEXES YIELD name, labelsOrTypes, state")
                indexes = list(result)
                
                online_count = sum(1 for r in indexes if r['state'] == 'ONLINE')
                
                for rec in indexes:
                    print(f"- Index: {rec['name']} | Label: {rec['labelsOrTypes']} | Status: {rec['state']}")
                
                if online_count >= 2:
                    print(f"\n✅ SUCCESS: Both Graph and Chunk indexes are ONLINE in '{db_name}'.")
                    return
                
                print(f"Waiting for indexes to populate (Attempt {i+1}/5)...")
                
        except Exception as e:
            print(f"❌ DATABASE ERROR: {e}")
        finally:
            driver.close()

if __name__ == "__main__":
    setup_indexes()








import os
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    NEO4J_URI = os.environ["NEO4J_URI"]
    NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
    NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
    NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
    ACADEMIC_API_KEY = os.environ["ACADEMIC_API_KEY"]

    CYPHER_MODEL = "gpt-oss-120b"
    QA_MODEL = "gpt-oss-120b"
    BASE_URL = "https://chat-ai.academiccloud.de/v1"

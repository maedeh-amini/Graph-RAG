import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key=os.environ["82a8404ad2c23c26bb2ced719964b19a"],  # set this first
    base_url="https://chat-ai.academiccloud.de/v1",
    model="meta-llama-3.1-8b-instruct",
    temperature=0
)

llm.invoke("How tall is the Eiffel tower?")

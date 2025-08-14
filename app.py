import os
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
import asyncio
import chromadb
# import logging

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ["GOOGLE_API_KEY"] = "AIzaSyCX8Kr5Xj1dutjeClYQ-fFN6GH6NTP_PLg"

# Settings control global defaults
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    embed_batch_size=100
)
Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
)

# Create the document index
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("test")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

if chroma_collection.count() == 0:
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
else:
    index = VectorStoreIndex.from_vector_store(
        vector_store,
    )

# Create a RAG tool using LlamaIndex
query_engine = index.as_query_engine(
    # we can optionally override the llm here
    # llm=Settings.llm,
)


async def search_documents(query: str) -> str:
    """Useful for answering natural language questions."""
    response = await query_engine.aquery(query)
    return str(response)


# Create an enhanced workflow with tool
agent = AgentWorkflow.from_tools_or_functions(
    [search_documents],
    llm=Settings.llm,
    system_prompt="""You are a helpful assistant that can search through documents to answer questions.""",
)


# Now we can ask questions about the documents or do calculations
async def main():
    response = await agent.run(
        "Оглавление"
    )
    print(response)


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())

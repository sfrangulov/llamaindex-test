import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import asyncio
import chromadb

# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(
    model_name="Qwen/Qwen3-Embedding-0.6B")
Settings.llm = Ollama(
    model="qwen3:0.6b",
    request_timeout=360.0,
    # Manually set the context window to limit memory usage
    context_window=8000,
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
        "Какие разделы не обязательны для заполнения в функциональной спецификации? Используй только информацию из документов, не основывайся на собственных знаниях."
    )
    print(response)


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())

import os
import sys
import asyncio
from typing import Any, Dict, List, Optional

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.schema import MetadataMode
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
from llama_index.core.postprocessor import (
    LongContextReorder,
    SimilarityPostprocessor,
    SentenceTransformerRerank,
)
# Try multiple import paths for cross-encoder reranker depending on LI version
try:
    # Modern path
    from llama_index.postprocessor import SentenceTransformerRerank  # type: ignore
except Exception:
    try:
        # Legacy path
        from llama_index.postprocessor.sentence_transformer_rerank import (  # type: ignore
            SentenceTransformerRerank,
        )
    except Exception:  # pragma: no cover - optional dependency
        SentenceTransformerRerank = None  # type: ignore

import chromadb
 
os.environ["GOOGLE_API_KEY"] = "AIzaSyCX8Kr5Xj1dutjeClYQ-fFN6GH6NTP_PLg"

# Configure API keys from environment (do not hardcode secrets)
if not os.environ.get("GOOGLE_API_KEY"):
    # Fail fast with a clear message; user must export GOOGLE_API_KEY
    print("ERROR: GOOGLE_API_KEY is not set. Export it in your shell before running.")
    # Avoid raising during import when used as a library; only exit if run as script
    if __name__ == "__main__":
        sys.exit(1)

# Settings control global defaults
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    embed_batch_size=100,
)
Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
    temperature=0.1,  # deterministic to reduce hallucinations
)
Settings.node_parser = SentenceSplitter(chunk_size=700, chunk_overlap=120)

# Create the document index
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("test")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def infer_language(text: str) -> str:
    # Simple heuristic: Cyrillic -> ru, else en
    return "ru" if any("\u0400" <= ch <= "\u04FF" for ch in text) else "en"


if chroma_collection.count() == 0:
    documents = SimpleDirectoryReader("data").load_data()
    # enrich metadata for filtering
    for d in documents:
        d.metadata.setdefault("file_name", d.metadata.get("file_name") or d.metadata.get("filename") or "unknown")
        d.metadata.setdefault("section", "unknown")
        d.metadata.setdefault("version", "v1")
        if "lang" not in d.metadata:
            d.metadata["lang"] = infer_language(d.text or "")

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
else:
    index = VectorStoreIndex.from_vector_store(vector_store)

from llama_index.core.vector_stores.types import VectorStoreQueryMode

# Base retriever (attempt hybrid if supported by the vector store)
retriever = index.as_retriever(
    similarity_top_k=40,
    vector_store_query_mode=VectorStoreQueryMode.HYBRID,
    alpha=0.35,  # boost keyword/sparse; ignored if store doesn't support hybrid
    sparse_top_k=40,
    hybrid_top_k=40,
)

node_postprocessors = []

# Cross-encoder rerank first for precision
try:
    if SentenceTransformerRerank is not None:
        node_postprocessors.append(
            SentenceTransformerRerank(
                top_n=12, model="cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        )
except Exception:
    pass

# Light or no similarity cutoff to avoid over-pruning
node_postprocessors.append(SimilarityPostprocessor(similarity_cutoff=0.05))

# Place salient chunks near the beginning/end for long prompts
node_postprocessors.append(LongContextReorder())

base_query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=node_postprocessors,
)

# Query transforms: HyDE (+ optional user paraphrase in future)
hyde = HyDEQueryTransform(include_original=True)
query_engine = TransformQueryEngine(base_query_engine, hyde)


async def search_documents(
    query: str,
    *,
    file_name: Optional[str] = None,
    section: Optional[str] = None,
    lang: Optional[str] = None,
    version: Optional[str] = None,
) -> str:
    """Search documents and return answer with citations.

    Optional metadata filters can be provided to narrow the search.
    In production, prefer to return a structured object; here we JSON-encode for the agent.
    """
    filters: List[ExactMatchFilter] = []
    if file_name:
        filters.append(ExactMatchFilter(key="file_name", value=file_name))
    if section:
        filters.append(ExactMatchFilter(key="section", value=section))
    if lang:
        filters.append(ExactMatchFilter(key="lang", value=lang))
    if version:
        filters.append(ExactMatchFilter(key="version", value=version))

    engine = query_engine
    if filters:
        # Rebuild retriever with metadata filters
        filtered_retriever = index.as_retriever(
            similarity_top_k=40,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            alpha=0.35,
            sparse_top_k=40,
            hybrid_top_k=40,
            filters=MetadataFilters(filters=filters),
        )
        filtered_engine = RetrieverQueryEngine(
            retriever=filtered_retriever,
            node_postprocessors=node_postprocessors,
        )
        engine = TransformQueryEngine(filtered_engine, hyde)

    response = await engine.aquery(query)

    # Build citation structure
    sources = []
    for sn in getattr(response, "source_nodes", []) or []:
        meta = sn.node.metadata or {}
        sources.append(
            {
                "score": sn.score,
                "file_name": meta.get("file_name") or meta.get("filename"),
                "section": meta.get("section"),
                "lang": meta.get("lang"),
                "version": meta.get("version"),
                "window": meta.get("window"),
                "text": sn.node.get_content(metadata_mode=MetadataMode.NONE),
            }
        )

    import json

    payload: Dict[str, Any] = {
        "answer": str(response),
        "sources": sources,
    }
    return json.dumps(payload, ensure_ascii=False)


# Create an enhanced workflow with tool
agent = AgentWorkflow.from_tools_or_functions(
    [search_documents],
    llm=Settings.llm,
    system_prompt=(
        "Ты — ассистент для поиска по локальным документам. "
        "Всегда используй инструмент search_documents для ответа и не отвечай без него. "
        "Отвечай кратко, опираясь только на найденные фрагменты. Если результатов нет — скажи, что не знаешь. "
        "В ответе используй точные цитаты и добавляй ссылки на источники из результатов инструмента."
    ),
)


# Now we can ask questions about the documents or do calculations
async def main():
    # Allow passing a custom query via CLI; default to a sensible demo query
    query = sys.argv[1] if len(sys.argv) > 1 else "Предмет разработки"

    # First, try via the agent (function-calling should invoke the tool)
    try:
        agent_resp = await agent.run(query, max_iterations=2)
        text = str(agent_resp)
        if text.strip() and text.strip().lower() != "я не знаю.":
            print(text)
            return
    except Exception:
        pass

    # Fallback: call the retrieval tool directly for a guaranteed answer with citations
    result_json = await search_documents(query)
    print(result_json)


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import concurrent.futures
import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from tqdm import tqdm

load_dotenv()


def _get_transformer_graph():
    llm = ChatOpenAI(
        temperature=0,
        model_name=os.getenv("OPENAI_GENERATIVE_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    return LLMGraphTransformer(llm=llm)


async def _extract_graphs_async(
    transformer: LLMGraphTransformer,
    documents: List[Document],
    max_concurrency: int = 5,
    show_progress: bool = True,
) -> List[GraphDocument]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _process_one(doc: Document, pbar: Optional[tqdm]):
        async with semaphore:
            try:
                res = await transformer.aprocess_response(doc)
            except Exception as e:
                doc_title = doc.metadata.get("scheme_title", "Unknown")
                print(f"\n[Warning] Error processing document '{doc_title}': {e}")
                res = None
            if pbar:
                pbar.update(1)
            return res

    pbar = (
        tqdm(total=len(documents), desc="Extracting entity graphs", unit="doc")
        if show_progress
        else None
    )
    try:
        tasks = [_process_one(doc, pbar) for doc in documents]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    finally:
        if pbar:
            pbar.close()


def get_entity_graphs(
    documents: List[Document],
    max_concurrency: int = 5,
    show_progress: bool = True,
) -> List[GraphDocument]:
    """
    Extract knowledge graph entities and relationships from documents concurrently.
    
    Args:
        documents: List of Document chunks to process.
        max_concurrency: Maximum number of concurrent OpenAI requests (default: 5).
        show_progress: Whether to display a tqdm progress bar (default: True).
    """
    if not documents:
        return []

    transformer = _get_transformer_graph()

    # Support environments with or without an existing event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run,
                _extract_graphs_async(
                    transformer, documents, max_concurrency, show_progress
                ),
            ).result()
    else:
        return asyncio.run(
            _extract_graphs_async(
                transformer, documents, max_concurrency, show_progress
            )
        )
from typing import List
import logging
import vertexai
from vertexai import rag
from google.api_core import retry

logger = logging.getLogger("vertex-rag-query")
logging.basicConfig(level=logging.INFO)

# ============================================================
# CONFIG
# ============================================================
PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"

# Retry config
RAG_RETRY = retry.Retry(
    predicate=retry.if_exception_type(),
    initial=1.0,
    maximum=10.0,
    multiplier=2.0,
    deadline=120.0,
)

# ============================================================
# HELPER: GET CORPUS BY DISPLAY NAME
# ============================================================
def get_corpus_by_display_name(display_name: str) -> str:
    """
    Returns the corpus resource name for a given display name.
    Raises an error if not found.
    """
    logger.info(f"Searching for corpus with display name: {display_name}")
    
    corpora = rag.list_corpora()
    for corpus in corpora:
        if corpus.display_name == display_name:
            logger.info(f"Found corpus: {corpus.name}")
            return corpus.name

    raise ValueError(f"RAG corpus with display name '{display_name}' not found")


# ============================================================
# RAG QUERY FUNCTION (INDEPENDENT)
# ============================================================
def query_rag_corpus(
    query: str,
    corpus_name: str = None,
    top_k: int = 5,
) -> List[dict]:
    """
    Queries a RAG corpus and returns retrieved chunks.

    Args:
        query: User query text
        corpus_name: Name or display name of the corpus
        top_k: Number of chunks to retrieve

    Returns:
        List of retrieved passages with metadata
    """
    logger.info("------------------------------------------------------------")
    logger.info("Entering query_rag_corpus()")
    logger.info(f"Query: {query}")
    logger.info(f"Top K: {top_k}")

    # Initialize Vertex AI if not already initialized
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        logger.info("Vertex AI initialized")
    except Exception as e:
        logger.warning(f"Vertex AI may already be initialized: {e}")

    # Resolve corpus name if a display name is given
    if corpus_name is None:
        raise ValueError("corpus_name must be provided")

    # If user passed a display name instead of full resource name
    if not corpus_name.startswith("projects/"):
        corpus_name = get_corpus_by_display_name(corpus_name)

    retrieval_config = rag.RagRetrievalConfig(top_k=top_k)
    logger.info("Sending retrieval request to Vertex AI RAG")

    response = RAG_RETRY(rag.retrieve)(
        corpus_name=corpus_name,
        query=query,
        retrieval_config=retrieval_config,
    )

    logger.info(f"Retrieved {len(response.contexts)} contexts")
    logger.info("Exiting query_rag_corpus()")
    logger.info("------------------------------------------------------------")

    results = []
    for ctx in response.contexts:
        results.append({
            "text": ctx.text,
            "source_uri": ctx.source_uri,
            "score": ctx.score,
        })

    return results
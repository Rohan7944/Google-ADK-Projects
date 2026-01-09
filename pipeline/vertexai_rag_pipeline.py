import logging
import time
from typing import List

import vertexai
from vertexai import rag
from google.api_core import retry

# ------------------
# Configure Logging
# ------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s"
)
logger = logging.getLogger("vertex-rag-pipeline")

# ------------------
# Config Variables
# ------------------
PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"    # or your region
CORPUS_DISPLAY_NAME = "MyVertexRagCorpus"
GCS_PATHS: List[str] = [
    "gs://bucketname/file1.pdf",
    "gs://bucketname/file2.txt",
]

# Retry policy for create/import
RAG_RETRY = retry.Retry(
    predicate=retry.if_exception_type(),
    initial=1.0,
    maximum=10.0,
    multiplier=2.0,
    deadline=120.0,
)

# -------------
# Initialize Vertex AI
# -------------
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --------------------------------
# 1. Create a RAG Corpus
# --------------------------------
def create_rag_corpus(display_name: str):
    logger.info(f"Creating RAG Corpus: {display_name}")
    backend_config = rag.RagVectorDbConfig(
        # Use default managed DB
        rag_embedding_model_config=rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model="publishers/google/models/text-embedding-005"
            )
        )
    )

    # Create the corpus
    corpus = RAG_RETRY(rag.create_corpus)(
        display_name=display_name,
        backend_config=backend_config,
    )

    logger.info(f"Created Corpus: {corpus.name}")
    return corpus

# --------------------------------
# 2. Import Data to RAG Corpus
# --------------------------------
def import_to_rag_corpus(corpus_name: str, paths: List[str]):
    logger.info(f"Starting import of {len(paths)} files into {corpus_name}")

    # Optional transformation for chunking
    trans_config = rag.TransformationConfig(
        rag.ChunkingConfig(chunk_size=512, chunk_overlap=64)
    )

    response = RAG_RETRY(rag.import_files)(
        corpus_name,
        paths,
        transformation_config=trans_config,
    )

    logger.info(
        f"Import Completed: Imported={response.imported_rag_files_count}, "
        f"Skipped={response.skipped_rag_files_count}"
    )
    return response

# --------------------------------
# 3. Main Pipeline
# --------------------------------
def run_pipeline():
    try:
        corpus = create_rag_corpus(CORPUS_DISPLAY_NAME)
        
        # Pause briefly
        time.sleep(4)

        logger.info("=== Importing documents from GCS ===")
        import_response = import_to_rag_corpus(corpus.name, GCS_PATHS)

        logger.info("RAG pipeline finished successfully!")
    except Exception as err:
        logger.exception(f"Pipeline failed: {err}")
        raise

if __name__ == "__main__":
    run_pipeline()

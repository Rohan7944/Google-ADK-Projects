import logging
import time
from typing import List

import vertexai
from vertexai import rag
from google.api_core import retry

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
)

logger = logging.getLogger("vertex-rag-pipeline")

logger.info("============================================================")
logger.info("Vertex AI RAG Pipeline - Process Started")
logger.info("============================================================")

# ============================================================
# VARIABLE INITIALIZATION
# ============================================================
logger.info("Initializing configuration variables")

PROJECT_ID = "your-gcp-project-id"
logger.info(f"PROJECT_ID set to: {PROJECT_ID}")

LOCATION = "us-central1"
logger.info(f"LOCATION set to: {LOCATION}")

CORPUS_DISPLAY_NAME = "MyVertexRagCorpus"
logger.info(f"CORPUS_DISPLAY_NAME set to: {CORPUS_DISPLAY_NAME}")

GCS_PATHS: List[str] = [
    "gs://bucketname/file1.pdf",
    "gs://bucketname/file2.txt",
]
logger.info(f"GCS_PATHS initialized with {len(GCS_PATHS)} paths")
for path in GCS_PATHS:
    logger.info(f"  └── GCS_PATH: {path}")

# ============================================================
# RETRY CONFIGURATION
# ============================================================
logger.info("Configuring retry policy for RAG operations")

RAG_RETRY = retry.Retry(
    predicate=retry.if_exception_type(),
    initial=1.0,
    maximum=10.0,
    multiplier=2.0,
    deadline=120.0,
)

logger.info(
    "Retry policy configured | "
    "initial=1.0s, maximum=10.0s, multiplier=2.0, deadline=120.0s"
)

# ============================================================
# VERTEX AI INITIALIZATION
# ============================================================
logger.info("Initializing Vertex AI SDK")
logger.info(f"Calling vertexai.init(project={PROJECT_ID}, location={LOCATION})")

vertexai.init(project=PROJECT_ID, location=LOCATION)

logger.info("Vertex AI SDK initialization completed successfully")

# ============================================================
# RAG CORPUS CREATION
# ============================================================
def create_rag_corpus(display_name: str):
    logger.info("------------------------------------------------------------")
    logger.info("Entering create_rag_corpus()")
    logger.info(f"Requested corpus display name: {display_name}")

    logger.info("Configuring embedding model for RAG corpus")
    embedding_config = rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
            publisher_model="publishers/google/models/text-embedding-005"
        )
    )
    logger.info("Embedding model set to: text-embedding-005")

    logger.info("Configuring vector database backend")
    backend_config = rag.RagVectorDbConfig(
        rag_embedding_model_config=embedding_config
    )

    logger.info("Sending request to Vertex AI to create RAG corpus")

    corpus = RAG_RETRY(rag.create_corpus)(
        display_name=display_name,
        backend_config=backend_config,
    )

    logger.info("RAG corpus creation request completed")
    logger.info(f"RAG Corpus Resource Name: {corpus.name}")

    logger.info("Exiting create_rag_corpus()")
    logger.info("------------------------------------------------------------")

    return corpus

# ============================================================
# RAG FILE IMPORT
# ============================================================
def import_to_rag_corpus(corpus_name: str, paths: List[str]):
    logger.info("------------------------------------------------------------")
    logger.info("Entering import_to_rag_corpus()")
    logger.info(f"Target corpus: {corpus_name}")
    logger.info(f"Number of files to import: {len(paths)}")

    logger.info("Configuring chunking and transformation settings")
    transformation_config = rag.TransformationConfig(
        rag.ChunkingConfig(
            chunk_size=512,
            chunk_overlap=64
        )
    )

    logger.info("Chunking configuration:")
    logger.info("  ├── chunk_size: 512")
    logger.info("  └── chunk_overlap: 64")

    logger.info("Submitting import_files() request to Vertex AI")

    response = RAG_RETRY(rag.import_files)(
        corpus_name,
        paths,
        transformation_config=transformation_config,
    )

    logger.info("Import operation completed successfully")
    logger.info(
        f"Import summary | "
        f"Imported: {response.imported_rag_files_count}, "
        f"Skipped: {response.skipped_rag_files_count}"
    )

    logger.info("Exiting import_to_rag_corpus()")
    logger.info("------------------------------------------------------------")

    return response

# ============================================================
# MAIN PIPELINE EXECUTION
# ============================================================
def run_pipeline():
    logger.info("============================================================")
    logger.info("RAG Pipeline Execution Started")
    logger.info("============================================================")

    try:
        logger.info("Step 1: Creating RAG corpus")
        corpus = create_rag_corpus(CORPUS_DISPLAY_NAME)

        logger.info("Step 1 completed successfully")
        logger.info("Sleeping for 4 seconds to allow backend stabilization")
        time.sleep(4)

        logger.info("Step 2: Importing documents from Google Cloud Storage")
        import_to_rag_corpus(corpus.name, GCS_PATHS)

        logger.info("Step 2 completed successfully")

        logger.info("============================================================")
        logger.info("RAG Pipeline Execution Finished Successfully")
        logger.info("============================================================")

    except Exception as error:
        logger.error("============================================================")
        logger.error("RAG Pipeline Execution FAILED")
        logger.error("============================================================")
        logger.exception(error)
        raise

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    logger.info("Script invoked directly (__main__)")
    run_pipeline()
"""
posts_rag2 - Aurora to OpenSearch Ingestion Lambda

This Lambda function ingests Facebook post data with pre-computed embeddings
from Aurora PostgreSQL into OpenSearch for a RAG pipeline.
"""

import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Any

import boto3
import psycopg2
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from requests_aws4auth import AWS4Auth


# Configuration from environment variables
class Config:
    # Aurora PostgreSQL
    AURORA_HOST = os.environ.get("AURORA_HOST")
    AURORA_PORT = int(os.environ.get("AURORA_PORT", "5432"))
    AURORA_USER = os.environ.get("AURORA_USER")
    AURORA_PASSWORD = os.environ.get("AURORA_PASSWORD")
    AURORA_DB_NAME = os.environ.get("AURORA_DB_NAME")
    AURORA_READ_BATCH_SIZE = int(os.environ.get("AURORA_READ_BATCH_SIZE", "1000"))

    # OpenSearch
    OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "")
    OPENSEARCH_REGION = os.environ.get("OPENSEARCH_REGION", "")
    INDEX_NAME = "rag-posts"

    # Index settings
    NUMBER_OF_SHARDS = int(os.environ.get("NUMBER_OF_SHARDS", "3"))
    NUMBER_OF_REPLICAS = int(os.environ.get("NUMBER_OF_REPLICAS", "1"))
    REFRESH_INTERVAL = os.environ.get("REFRESH_INTERVAL", "1s")

    # Bulk ingestion
    BULK_BATCH_DOCS = int(os.environ.get("BULK_BATCH_DOCS", "1000"))
    BULK_MAX_RETRIES = int(os.environ.get("BULK_MAX_RETRIES", "3"))

    # Schema validation
    REQUIRE_STRICT_SCHEMA = os.environ.get("REQUIRE_STRICT_SCHEMA", "false").lower() == "true"

    # Logging
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")


# Configure structured JSON logging
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "extra"):
            log_record.update(record.extra)
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add JSON handler
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    return logger


logger = setup_logging()


# CloudWatch metrics helper
class MetricsPublisher:
    def __init__(self):
        self.cloudwatch = boto3.client("cloudwatch")
        self.namespace = "posts_rag2"

    def put_metric(self, metric_name: str, value: float, unit: str = "Count"):
        try:
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        "MetricName": metric_name,
                        "Value": value,
                        "Unit": unit,
                        "Timestamp": datetime.now(timezone.utc),
                    }
                ],
            )
        except Exception as e:
            logger.warning(f"Failed to publish metric {metric_name}: {e}")


metrics = MetricsPublisher()


def get_opensearch_client() -> OpenSearch:
    """Create OpenSearch client with AWS SigV4 authentication."""
    credentials = boto3.Session().get_credentials()
    auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        Config.OPENSEARCH_REGION,
        "es",
        session_token=credentials.token,
    )

    # Parse host from URL
    host = Config.OPENSEARCH_HOST.replace("https://", "").replace("http://", "")

    client = OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )

    return client


def get_aurora_connection():
    """Create Aurora PostgreSQL connection."""
    return psycopg2.connect(
        host=Config.AURORA_HOST,
        port=Config.AURORA_PORT,
        user=Config.AURORA_USER,
        password=Config.AURORA_PASSWORD,
        dbname=Config.AURORA_DB_NAME,
    )


def get_index_settings() -> dict:
    """Return OpenSearch index settings."""
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": Config.NUMBER_OF_SHARDS,
                "number_of_replicas": Config.NUMBER_OF_REPLICAS,
                "refresh_interval": Config.REFRESH_INTERVAL,
            }
        },
        "mappings": {
            "properties": {
                "post_id": {"type": "keyword"},
                "timestamp": {"type": "date"},
                "title": {"type": "text", "analyzer": "icu_analyzer"},
                "full_chunk": {"type": "text", "analyzer": "icu_analyzer"},
                "has_consultant": {"type": "boolean"},
                "engagement_norm": {"type": "float"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosine",
                    },
                },
            }
        },
    }


def ensure_index_exists(client: OpenSearch) -> bool:
    """
    Ensure the OpenSearch index exists with correct configuration.
    Returns True if index is ready, False if there's a schema mismatch.
    """
    index_name = Config.INDEX_NAME

    if client.indices.exists(index=index_name):
        logger.info(f"Index '{index_name}' already exists, validating schema")

        # Get current index settings and mappings
        settings = client.indices.get_settings(index=index_name)
        mappings = client.indices.get_mapping(index=index_name)

        # Validate knn setting
        knn_enabled = settings.get(index_name, {}).get("settings", {}).get("index", {}).get("knn", "false")
        if str(knn_enabled).lower() != "true":
            msg = f"Index '{index_name}' does not have knn enabled"
            if Config.REQUIRE_STRICT_SCHEMA:
                raise ValueError(msg)
            logger.warning(msg)

        # Validate embedding dimension
        embedding_props = (
            mappings.get(index_name, {})
            .get("mappings", {})
            .get("properties", {})
            .get("embedding", {})
        )
        dimension = embedding_props.get("dimension")
        if dimension != 1024:
            msg = f"Index '{index_name}' embedding dimension is {dimension}, expected 1024"
            if Config.REQUIRE_STRICT_SCHEMA:
                raise ValueError(msg)
            logger.warning(msg)

        logger.info(f"Index '{index_name}' validation completed")
        return True

    # Create index
    logger.info(f"Creating index '{index_name}'")
    index_body = get_index_settings()
    client.indices.create(index=index_name, body=index_body)
    logger.info(f"Index '{index_name}' created successfully")

    return True


def validate_record(record: dict) -> tuple[bool, str]:
    """
    Validate a single record.
    Returns (is_valid, error_message).
    """
    required_fields = ["post_id", "timestamp", "title", "full_chunk", "has_consultant", "engagement_norm", "embedding"]

    # Check required fields
    for field in required_fields:
        if field not in record or record[field] is None:
            return False, f"Missing required field: {field}"

    # Validate embedding
    embedding = record["embedding"]
    if not isinstance(embedding, (list, tuple)):
        return False, "Embedding is not an array"

    if len(embedding) != 1024:
        return False, f"Embedding has {len(embedding)} dimensions, expected 1024"

    # Check all values are numeric
    for i, val in enumerate(embedding):
        if not isinstance(val, (int, float)):
            return False, f"Embedding value at index {i} is not numeric"

    return True, ""


def stream_records_from_aurora(conn, batch_size: int):
    """
    Stream records from Aurora PostgreSQL in batches.
    Only selects records where ingested_at IS NULL.
    """
    query = """
        SELECT
            post_id,
            timestamp,
            title,
            full_chunk,
            has_consultant,
            engagement_norm,
            embedding
        FROM facebook_embeddings
        WHERE ingested_at IS NULL
        ORDER BY post_id
    """

    with conn.cursor(name="fetch_posts") as cursor:
        cursor.itersize = batch_size
        cursor.execute(query)

        batch = []
        for row in cursor:
            record = {
                "post_id": row[0],
                "timestamp": row[1],
                "title": row[2],
                "full_chunk": row[3],
                "has_consultant": row[4],
                "engagement_norm": row[5],
                "embedding": row[6],
            }
            batch.append(record)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


def update_ingested_at(conn, post_ids: list[str]):
    """Update ingested_at timestamp for successfully processed records."""
    if not post_ids:
        return

    with conn.cursor() as cursor:
        # Use ANY for efficient batch update
        cursor.execute(
            """
            UPDATE facebook_embeddings
            SET ingested_at = NOW()
            WHERE post_id = ANY(%s)
            """,
            (post_ids,),
        )
    conn.commit()


def prepare_bulk_action(record: dict) -> dict:
    """Prepare a document for bulk indexing."""
    # Format timestamp for OpenSearch
    timestamp = record["timestamp"]
    if isinstance(timestamp, datetime):
        timestamp = timestamp.isoformat()

    return {
        "_index": Config.INDEX_NAME,
        "_id": record["post_id"],
        "_source": {
            "post_id": record["post_id"],
            "timestamp": timestamp,
            "title": record["title"],
            "full_chunk": record["full_chunk"],
            "has_consultant": record["has_consultant"],
            "engagement_norm": record["engagement_norm"],
            "embedding": record["embedding"],
        },
    }


def bulk_ingest_with_retry(
    client: OpenSearch, actions: list[dict], max_retries: int
) -> tuple[int, int, list[str]]:
    """
    Perform bulk ingestion with retry logic for failed items.
    Returns (success_count, fail_count, successful_ids).
    """
    successful_ids = []
    failed_actions = actions.copy()
    total_success = 0
    total_fail = 0
    retry_count = 0

    while failed_actions and retry_count <= max_retries:
        if retry_count > 0:
            # Exponential backoff with jitter
            base_delay = 2 ** retry_count
            jitter = random.uniform(0, 1)
            delay = base_delay + jitter
            logger.info(f"Retry {retry_count}/{max_retries}, waiting {delay:.2f}s")
            time.sleep(delay)

        try:
            success, errors = helpers.bulk(
                client,
                failed_actions,
                raise_on_error=False,
                raise_on_exception=False,
            )

            total_success += success

            # Collect failed actions for retry
            new_failed_actions = []
            if errors:
                failed_ids = set()
                for error in errors:
                    if "index" in error:
                        failed_id = error["index"].get("_id")
                        if failed_id:
                            failed_ids.add(failed_id)

                for action in failed_actions:
                    if action["_id"] in failed_ids:
                        new_failed_actions.append(action)
                    else:
                        successful_ids.append(action["_id"])
            else:
                # All succeeded
                for action in failed_actions:
                    successful_ids.append(action["_id"])

            failed_actions = new_failed_actions

        except Exception as e:
            logger.error(f"Bulk operation failed: {e}")
            retry_count += 1
            continue

        retry_count += 1

    total_fail = len(failed_actions)

    return total_success, total_fail, successful_ids


def process_batch(
    os_client: OpenSearch,
    aurora_conn,
    records: list[dict],
) -> dict:
    """
    Process a batch of records: validate, ingest, and update Aurora.
    Returns batch statistics.
    """
    start_time = time.time()

    valid_records = []
    invalid_count = 0

    # Validate records
    for record in records:
        is_valid, error_msg = validate_record(record)
        if is_valid:
            valid_records.append(record)
        else:
            invalid_count += 1
            logger.warning(
                f"Invalid record skipped",
                extra={
                    "extra": {
                        "post_id": record.get("post_id", "unknown"),
                        "error": error_msg,
                    }
                },
            )

    if not valid_records:
        return {
            "docs_sent": 0,
            "docs_succeeded": 0,
            "docs_failed": invalid_count,
            "retries": 0,
            "latency_ms": 0,
        }

    # Prepare bulk actions
    actions = [prepare_bulk_action(r) for r in valid_records]

    # Perform bulk ingestion with retry
    success_count, fail_count, successful_ids = bulk_ingest_with_retry(
        os_client, actions, Config.BULK_MAX_RETRIES
    )

    # Update ingested_at for successful records
    if successful_ids:
        update_ingested_at(aurora_conn, successful_ids)

    latency_ms = (time.time() - start_time) * 1000

    return {
        "docs_sent": len(valid_records),
        "docs_succeeded": success_count,
        "docs_failed": fail_count + invalid_count,
        "latency_ms": latency_ms,
    }


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Main Lambda handler.
    Ingests data from Aurora PostgreSQL to OpenSearch.
    """
    logger.info("Starting posts_rag2 ingestion")

    # Validate configuration
    required_config = [
        "AURORA_HOST",
        "AURORA_USER",
        "AURORA_PASSWORD",
        "AURORA_DB_NAME",
        "OPENSEARCH_HOST",
        "OPENSEARCH_REGION",
    ]
    missing = [c for c in required_config if not getattr(Config, c)]
    if missing:
        error_msg = f"Missing required configuration: {missing}"
        logger.error(error_msg)
        return {"statusCode": 500, "body": json.dumps({"error": error_msg})}

    try:
        # Initialize clients
        os_client = get_opensearch_client()
        aurora_conn = get_aurora_connection()

        # Ensure index exists
        ensure_index_exists(os_client)

        # Process records
        total_stats = {
            "batches_processed": 0,
            "docs_ingested": 0,
            "docs_failed": 0,
        }

        for batch in stream_records_from_aurora(aurora_conn, Config.AURORA_READ_BATCH_SIZE):
            # Process in smaller bulk batches if needed
            for i in range(0, len(batch), Config.BULK_BATCH_DOCS):
                chunk = batch[i : i + Config.BULK_BATCH_DOCS]
                batch_stats = process_batch(os_client, aurora_conn, chunk)

                total_stats["batches_processed"] += 1
                total_stats["docs_ingested"] += batch_stats["docs_succeeded"]
                total_stats["docs_failed"] += batch_stats["docs_failed"]

                # Log batch metrics
                logger.info(
                    "Batch completed",
                    extra={
                        "extra": {
                            "batch_number": total_stats["batches_processed"],
                            "docs_sent": batch_stats["docs_sent"],
                            "docs_succeeded": batch_stats["docs_succeeded"],
                            "docs_failed": batch_stats["docs_failed"],
                            "latency_ms": batch_stats["latency_ms"],
                        }
                    },
                )

                # Publish CloudWatch metrics
                metrics.put_metric("batches_processed", 1)
                metrics.put_metric("docs_ingested", batch_stats["docs_succeeded"])
                metrics.put_metric("docs_failed", batch_stats["docs_failed"])
                metrics.put_metric("bulk_latency_ms", batch_stats["latency_ms"], "Milliseconds")

        # Close Aurora connection
        aurora_conn.close()

        logger.info(
            "Ingestion completed",
            extra={"extra": total_stats},
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Ingestion completed successfully",
                    "stats": total_stats,
                }
            ),
        }

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }

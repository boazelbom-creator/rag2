# posts_rag2

AWS Lambda function that ingests Facebook post data with pre-computed embeddings from Aurora PostgreSQL into OpenSearch for a RAG (Retrieval-Augmented Generation) pipeline.

## Overview

This function is part of a data pipeline designed to enable hybrid semantic and lexical search over Hebrew-language social media content. It reads posts from Aurora PostgreSQL (processed by upstream `posts_rag1` Lambda) and indexes them into OpenSearch with k-NN vector search enabled.

**Note:** This Lambda is the authoritative source for OpenSearch index creation and configuration. The index settings (shards, replicas, analyzer, k-NN configuration) are defined in this codebase.

### Features

- Streams data from Aurora PostgreSQL in configurable batches
- Creates OpenSearch index with k-NN enabled and ICU analyzer for Hebrew text
- Validates records (embedding dimension, required fields)
- Bulk ingestion with exponential backoff retry logic
- Resume capability via `ingested_at` timestamp marker
- Idempotent indexing (document `_id` = `post_id`)
- Structured JSON logging
- CloudWatch metrics publishing

## Prerequisites

- Python 3.12
- AWS account with:
  - Aurora PostgreSQL database
  - OpenSearch domain
  - IAM role with appropriate permissions

## Setup

### 1. Create Virtual Environment

```bash
cd posts_rag2
python3.12 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Test Dependencies (for local testing)

```bash
pip install pytest boto3
```

## Configuration

Set the following environment variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AURORA_HOST` | Yes | - | Aurora PostgreSQL endpoint |
| `AURORA_PORT` | No | 5432 | Database port |
| `AURORA_USER` | Yes | - | Database username |
| `AURORA_PASSWORD` | Yes | - | Database password |
| `AURORA_DB_NAME` | Yes | - | Database name |
| `AURORA_READ_BATCH_SIZE` | No | 1000 | Records per database fetch |
| `OPENSEARCH_HOST` | Yes | - | OpenSearch domain endpoint (e.g., `https://search-xyz.region.es.amazonaws.com`) |
| `OPENSEARCH_REGION` | Yes | - | AWS region for SigV4 signing |
| `NUMBER_OF_SHARDS` | No | 3 | Index shard count |
| `NUMBER_OF_REPLICAS` | No | 1 | Index replica count |
| `REFRESH_INTERVAL` | No | 1s | Index refresh interval |
| `BULK_BATCH_DOCS` | No | 1000 | Documents per bulk request |
| `BULK_MAX_RETRIES` | No | 3 | Max retries for failed items |
| `REQUIRE_STRICT_SCHEMA` | No | false | Fail on schema mismatch if true |
| `LOG_LEVEL` | No | INFO | Logging verbosity |

## Testing

Run the test suite:

```bash
source venv/bin/activate
pytest test_lambda_function.py -v
```

## Deployment

### Package for Lambda

```bash
# Create package directory
mkdir -p package

# Install dependencies into package directory
pip install -r requirements.txt -t package/

# Copy Lambda function code
cp lambda_function.py package/

# Create deployment zip
cd package
zip -r ../posts_rag2.zip .
cd ..

# Verify package size (must be < 250 MB unzipped)
du -sh package/
```

### Lambda Configuration

| Setting | Recommended Value |
|---------|-------------------|
| Runtime | Python 3.12 |
| Handler | `lambda_function.lambda_handler` |
| Timeout | 15 minutes |
| Memory | 1024 MB minimum |
| VPC | Required for Aurora access |

### IAM Permissions

The Lambda execution role needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "es:ESHttpGet",
        "es:ESHttpHead",
        "es:ESHttpPost",
        "es:ESHttpPut"
      ],
      "Resource": "arn:aws:es:<region>:<account>:domain/<domain-name>/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData"
      ],
      "Resource": "*"
    }
  ]
}
```

## Source Data Schema

The function reads from the `facebook_embeddings` table:

| Field | Type | Description |
|-------|------|-------------|
| `post_id` | TEXT | Primary key |
| `timestamp` | TIMESTAMP | Post publication time |
| `title` | TEXT | Post title (normalized for BM25) |
| `full_chunk` | TEXT | Full post content (normalized for BM25) |
| `has_consultant` | BOOLEAN | Consultant involvement flag |
| `engagement_norm` | FLOAT | Normalized engagement score (0-1) |
| `embedding` | FLOAT8[] | 1024-dimensional vector |
| `ingested_at` | TIMESTAMP | Resume marker (NULL if not processed) |

## OpenSearch Index

Index name: `rag-posts`

### Mappings

| Field | Type | Analyzer |
|-------|------|----------|
| `post_id` | keyword | - |
| `timestamp` | date | - |
| `title` | text | icu_analyzer |
| `full_chunk` | text | icu_analyzer |
| `has_consultant` | boolean | - |
| `engagement_norm` | float | - |
| `embedding` | knn_vector (1024-dim, HNSW, cosinesimil, Lucene engine) | - |

## CloudWatch Metrics

Namespace: `posts_rag2`

| Metric | Unit | Description |
|--------|------|-------------|
| `batches_processed` | Count | Number of batches completed |
| `docs_ingested` | Count | Successfully indexed documents |
| `docs_failed` | Count | Documents that failed after retries |
| `bulk_latency_ms` | Milliseconds | Time per bulk operation |

## Query Phase (Future)

The index supports hybrid search with scoring formula:

```
final_score = 0.5 * vector_similarity
            + 0.2 * has_consultant
            + 0.2 * BM25_score
            + 0.1 * engagement_norm
```

When embedding queries, use `input_type: "search_query"` (not `"search_document"`):

```python
response = bedrock.invoke_model(
    modelId="cohere.embed-multilingual-v3",
    body=json.dumps({
        "texts": [user_query],
        "input_type": "search_query"
    })
)
```

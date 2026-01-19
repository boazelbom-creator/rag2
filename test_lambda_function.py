"""
Unit tests for posts_rag2 Lambda function.

Run with: pytest test_lambda_function.py -v
"""

import json
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from lambda_function import (
    Config,
    JsonFormatter,
    MetricsPublisher,
    get_index_settings,
    prepare_bulk_action,
    validate_record,
    lambda_handler,
)


class TestConfig:
    """Tests for the Config class."""

    def test_default_aurora_port(self):
        assert Config.AURORA_PORT == 5432

    def test_default_aurora_read_batch_size(self):
        assert Config.AURORA_READ_BATCH_SIZE == 1000

    def test_default_bulk_batch_docs(self):
        assert Config.BULK_BATCH_DOCS == 1000

    def test_default_bulk_max_retries(self):
        assert Config.BULK_MAX_RETRIES == 3

    def test_default_number_of_shards(self):
        assert Config.NUMBER_OF_SHARDS == 3

    def test_default_number_of_replicas(self):
        assert Config.NUMBER_OF_REPLICAS == 1

    def test_default_refresh_interval(self):
        assert Config.REFRESH_INTERVAL == "1s"

    def test_default_log_level(self):
        assert Config.LOG_LEVEL == "INFO"

    def test_default_require_strict_schema(self):
        assert Config.REQUIRE_STRICT_SCHEMA is False

    def test_index_name(self):
        assert Config.INDEX_NAME == "rag-posts"


class TestValidateRecord:
    """Tests for the validate_record function."""

    @pytest.fixture
    def valid_record(self):
        """Create a valid record for testing."""
        return {
            "post_id": "test123",
            "timestamp": datetime.now(),
            "title": "Test Title",
            "full_chunk": "Test content in Hebrew אבגדהו",
            "has_consultant": True,
            "engagement_norm": 0.75,
            "embedding": [0.1] * 1024,
        }

    def test_valid_record(self, valid_record):
        is_valid, error = validate_record(valid_record)
        assert is_valid is True
        assert error == ""

    def test_valid_record_with_integer_embedding(self, valid_record):
        valid_record["embedding"] = [1] * 1024  # integers are valid
        is_valid, error = validate_record(valid_record)
        assert is_valid is True

    def test_missing_post_id(self, valid_record):
        del valid_record["post_id"]
        is_valid, error = validate_record(valid_record)
        assert is_valid is False
        assert "post_id" in error

    def test_missing_timestamp(self, valid_record):
        del valid_record["timestamp"]
        is_valid, error = validate_record(valid_record)
        assert is_valid is False
        assert "timestamp" in error

    def test_missing_title(self, valid_record):
        del valid_record["title"]
        is_valid, error = validate_record(valid_record)
        assert is_valid is False
        assert "title" in error

    def test_missing_full_chunk(self, valid_record):
        del valid_record["full_chunk"]
        is_valid, error = validate_record(valid_record)
        assert is_valid is False
        assert "full_chunk" in error

    def test_missing_has_consultant(self, valid_record):
        del valid_record["has_consultant"]
        is_valid, error = validate_record(valid_record)
        assert is_valid is False
        assert "has_consultant" in error

    def test_missing_engagement_norm(self, valid_record):
        del valid_record["engagement_norm"]
        is_valid, error = validate_record(valid_record)
        assert is_valid is False
        assert "engagement_norm" in error

    def test_missing_embedding(self, valid_record):
        del valid_record["embedding"]
        is_valid, error = validate_record(valid_record)
        assert is_valid is False
        assert "embedding" in error

    def test_none_field_value(self, valid_record):
        valid_record["title"] = None
        is_valid, error = validate_record(valid_record)
        assert is_valid is False
        assert "title" in error

    def test_embedding_wrong_dimension_512(self, valid_record):
        valid_record["embedding"] = [0.1] * 512
        is_valid, error = validate_record(valid_record)
        assert is_valid is False
        assert "512" in error
        assert "1024" in error

    def test_embedding_wrong_dimension_2048(self, valid_record):
        valid_record["embedding"] = [0.1] * 2048
        is_valid, error = validate_record(valid_record)
        assert is_valid is False
        assert "2048" in error

    def test_embedding_empty(self, valid_record):
        valid_record["embedding"] = []
        is_valid, error = validate_record(valid_record)
        assert is_valid is False
        assert "0 dimensions" in error

    def test_embedding_not_array_string(self, valid_record):
        valid_record["embedding"] = "not_an_array"
        is_valid, error = validate_record(valid_record)
        assert is_valid is False
        assert "not an array" in error

    def test_embedding_not_array_dict(self, valid_record):
        valid_record["embedding"] = {"values": [0.1] * 1024}
        is_valid, error = validate_record(valid_record)
        assert is_valid is False
        assert "not an array" in error

    def test_embedding_non_numeric_value_string(self, valid_record):
        valid_record["embedding"] = [0.1] * 1023 + ["not_a_number"]
        is_valid, error = validate_record(valid_record)
        assert is_valid is False
        assert "not numeric" in error
        assert "1023" in error

    def test_embedding_non_numeric_value_none(self, valid_record):
        valid_record["embedding"] = [0.1] * 500 + [None] + [0.1] * 523
        is_valid, error = validate_record(valid_record)
        assert is_valid is False
        assert "not numeric" in error

    def test_embedding_tuple_is_valid(self, valid_record):
        valid_record["embedding"] = tuple([0.1] * 1024)
        is_valid, error = validate_record(valid_record)
        assert is_valid is True


class TestPrepareBulkAction:
    """Tests for the prepare_bulk_action function."""

    @pytest.fixture
    def sample_record(self):
        return {
            "post_id": "post_abc123",
            "timestamp": datetime(2024, 6, 15, 10, 30, 0),
            "title": "כותרת בעברית",
            "full_chunk": "תוכן הפוסט המלא בעברית",
            "has_consultant": True,
            "engagement_norm": 0.85,
            "embedding": [0.1, 0.2, 0.3] + [0.0] * 1021,
        }

    def test_index_name(self, sample_record):
        action = prepare_bulk_action(sample_record)
        assert action["_index"] == "rag-posts"

    def test_document_id_equals_post_id(self, sample_record):
        action = prepare_bulk_action(sample_record)
        assert action["_id"] == sample_record["post_id"]

    def test_source_contains_all_fields(self, sample_record):
        action = prepare_bulk_action(sample_record)
        expected_fields = [
            "post_id",
            "timestamp",
            "title",
            "full_chunk",
            "has_consultant",
            "engagement_norm",
            "embedding",
        ]
        for field in expected_fields:
            assert field in action["_source"]

    def test_timestamp_formatted_as_iso(self, sample_record):
        action = prepare_bulk_action(sample_record)
        assert action["_source"]["timestamp"] == "2024-06-15T10:30:00"

    def test_timestamp_string_passthrough(self, sample_record):
        sample_record["timestamp"] = "2024-06-15T10:30:00Z"
        action = prepare_bulk_action(sample_record)
        assert action["_source"]["timestamp"] == "2024-06-15T10:30:00Z"

    def test_hebrew_content_preserved(self, sample_record):
        action = prepare_bulk_action(sample_record)
        assert action["_source"]["title"] == "כותרת בעברית"
        assert action["_source"]["full_chunk"] == "תוכן הפוסט המלא בעברית"

    def test_boolean_preserved(self, sample_record):
        action = prepare_bulk_action(sample_record)
        assert action["_source"]["has_consultant"] is True

        sample_record["has_consultant"] = False
        action = prepare_bulk_action(sample_record)
        assert action["_source"]["has_consultant"] is False

    def test_embedding_length_preserved(self, sample_record):
        action = prepare_bulk_action(sample_record)
        assert len(action["_source"]["embedding"]) == 1024


class TestGetIndexSettings:
    """Tests for the get_index_settings function."""

    def test_returns_dict_with_settings_and_mappings(self):
        settings = get_index_settings()
        assert "settings" in settings
        assert "mappings" in settings

    def test_knn_enabled(self):
        settings = get_index_settings()
        assert settings["settings"]["index"]["knn"] is True

    def test_number_of_shards(self):
        settings = get_index_settings()
        assert settings["settings"]["index"]["number_of_shards"] == 3

    def test_number_of_replicas(self):
        settings = get_index_settings()
        assert settings["settings"]["index"]["number_of_replicas"] == 1

    def test_refresh_interval(self):
        settings = get_index_settings()
        assert settings["settings"]["index"]["refresh_interval"] == "1s"

    def test_post_id_is_keyword(self):
        settings = get_index_settings()
        assert settings["mappings"]["properties"]["post_id"]["type"] == "keyword"

    def test_timestamp_is_date(self):
        settings = get_index_settings()
        assert settings["mappings"]["properties"]["timestamp"]["type"] == "date"

    def test_title_uses_icu_analyzer(self):
        settings = get_index_settings()
        title_props = settings["mappings"]["properties"]["title"]
        assert title_props["type"] == "text"
        assert title_props["analyzer"] == "icu_analyzer"

    def test_full_chunk_uses_icu_analyzer(self):
        settings = get_index_settings()
        full_chunk_props = settings["mappings"]["properties"]["full_chunk"]
        assert full_chunk_props["type"] == "text"
        assert full_chunk_props["analyzer"] == "icu_analyzer"

    def test_has_consultant_is_boolean(self):
        settings = get_index_settings()
        assert settings["mappings"]["properties"]["has_consultant"]["type"] == "boolean"

    def test_engagement_norm_is_float(self):
        settings = get_index_settings()
        assert settings["mappings"]["properties"]["engagement_norm"]["type"] == "float"

    def test_embedding_is_knn_vector(self):
        settings = get_index_settings()
        embedding_props = settings["mappings"]["properties"]["embedding"]
        assert embedding_props["type"] == "knn_vector"

    def test_embedding_dimension_is_1024(self):
        settings = get_index_settings()
        embedding_props = settings["mappings"]["properties"]["embedding"]
        assert embedding_props["dimension"] == 1024

    def test_embedding_uses_hnsw(self):
        settings = get_index_settings()
        embedding_props = settings["mappings"]["properties"]["embedding"]
        assert embedding_props["method"]["name"] == "hnsw"

    def test_embedding_uses_cosine(self):
        settings = get_index_settings()
        embedding_props = settings["mappings"]["properties"]["embedding"]
        assert embedding_props["method"]["space_type"] == "cosine"


class TestJsonFormatter:
    """Tests for the JsonFormatter class."""

    @pytest.fixture
    def formatter(self):
        return JsonFormatter()

    def test_output_is_valid_json(self, formatter):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_contains_timestamp(self, formatter):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "timestamp" in parsed

    def test_contains_level(self, formatter):
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "WARNING"

    def test_contains_message(self, formatter):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="My test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "My test message"

    def test_contains_logger_name(self, formatter):
        record = logging.LogRecord(
            name="my_logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["logger"] == "my_logger"

    def test_extra_fields_included(self, formatter):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.extra = {"batch_number": 5, "docs_succeeded": 1000}
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["batch_number"] == 5
        assert parsed["docs_succeeded"] == 1000


class TestLambdaHandler:
    """Tests for the lambda_handler function."""

    def test_missing_configuration_returns_500(self):
        result = lambda_handler({}, None)
        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert "Missing required configuration" in body["error"]

    def test_missing_configuration_lists_missing_vars(self):
        result = lambda_handler({}, None)
        body = json.loads(result["body"])
        assert "AURORA_HOST" in body["error"]
        assert "AURORA_USER" in body["error"]
        assert "AURORA_PASSWORD" in body["error"]
        assert "AURORA_DB_NAME" in body["error"]
        assert "OPENSEARCH_HOST" in body["error"]
        assert "OPENSEARCH_REGION" in body["error"]

    @patch("lambda_function.get_aurora_connection")
    @patch("lambda_function.get_opensearch_client")
    @patch("lambda_function.ensure_index_exists")
    @patch("lambda_function.stream_records_from_aurora")
    @patch.object(Config, "AURORA_HOST", "test-host")
    @patch.object(Config, "AURORA_USER", "test-user")
    @patch.object(Config, "AURORA_PASSWORD", "test-pass")
    @patch.object(Config, "AURORA_DB_NAME", "test-db")
    @patch.object(Config, "OPENSEARCH_HOST", "https://test-opensearch")
    @patch.object(Config, "OPENSEARCH_REGION", "us-east-1")
    def test_successful_ingestion_returns_200(
        self,
        mock_stream,
        mock_ensure_index,
        mock_os_client,
        mock_aurora_conn,
    ):
        # Setup mocks
        mock_stream.return_value = iter([])  # No records
        mock_aurora_conn.return_value = MagicMock()

        result = lambda_handler({}, None)

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert "stats" in body
        assert body["stats"]["batches_processed"] == 0
        assert body["stats"]["docs_ingested"] == 0


class TestMetricsPublisher:
    """Tests for the MetricsPublisher class."""

    @patch("boto3.client")
    def test_put_metric_calls_cloudwatch(self, mock_boto_client):
        mock_cw = MagicMock()
        mock_boto_client.return_value = mock_cw

        publisher = MetricsPublisher()
        publisher.put_metric("test_metric", 100, "Count")

        mock_cw.put_metric_data.assert_called_once()
        call_args = mock_cw.put_metric_data.call_args
        assert call_args[1]["Namespace"] == "posts_rag2"
        assert call_args[1]["MetricData"][0]["MetricName"] == "test_metric"
        assert call_args[1]["MetricData"][0]["Value"] == 100

    @patch("boto3.client")
    def test_put_metric_handles_exception(self, mock_boto_client):
        mock_cw = MagicMock()
        mock_cw.put_metric_data.side_effect = Exception("CloudWatch error")
        mock_boto_client.return_value = mock_cw

        publisher = MetricsPublisher()
        # Should not raise exception
        publisher.put_metric("test_metric", 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

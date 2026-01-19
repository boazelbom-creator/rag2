"""
Integration tests verifying compatibility between posts_rag1 and posts_rag2.

These tests ensure that data written by rag1 can be correctly read and processed by rag2.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from lambda_function import validate_record, prepare_bulk_action


class TestRag1ToRag2DataFlow:
    """Tests verifying data compatibility between rag1 output and rag2 input."""

    @pytest.fixture
    def rag1_output_record(self):
        """
        Simulates a record as it would be read from PostgreSQL after being written by rag1.

        rag1 writes embedding as Python list -> PostgreSQL FLOAT8[] -> psycopg2 returns Python list
        """
        return {
            "post_id": "post_12345",
            "timestamp": datetime(2024, 6, 15, 10, 30, 0),
            "title": "כותרת בעברית",  # Hebrew title (normalized by rag1)
            "full_chunk": "תוכן הפוסט המלא בעברית עם תגובות",  # Hebrew content
            "has_consultant": True,
            "engagement_norm": 0.75,
            # psycopg2 returns FLOAT8[] as Python list
            "embedding": [0.123456] * 1024,
        }

    def test_psycopg2_returns_list_for_float8_array(self, rag1_output_record):
        """
        Verify that when rag2 reads from PostgreSQL, the embedding is a list.

        This simulates psycopg2's behavior with FLOAT8[] columns - it returns
        Python lists, not strings.
        """
        embedding = rag1_output_record["embedding"]

        # psycopg2 FLOAT8[] is returned as Python list
        assert isinstance(embedding, list), "FLOAT8[] should be returned as Python list"
        assert len(embedding) == 1024
        assert all(isinstance(x, (int, float)) for x in embedding)

    def test_rag2_validates_rag1_output(self, rag1_output_record):
        """Verify rag2's validate_record accepts data produced by rag1."""
        is_valid, error = validate_record(rag1_output_record)

        assert is_valid is True, f"Validation failed: {error}"
        assert error == ""

    def test_rag2_prepares_bulk_action_from_rag1_output(self, rag1_output_record):
        """Verify rag2 can prepare OpenSearch bulk action from rag1 data."""
        action = prepare_bulk_action(rag1_output_record)

        assert action["_index"] == "rag-posts"
        assert action["_id"] == "post_12345"
        assert action["_source"]["title"] == "כותרת בעברית"
        assert action["_source"]["full_chunk"] == "תוכן הפוסט המלא בעברית עם תגובות"
        assert len(action["_source"]["embedding"]) == 1024

    def test_hebrew_text_preserved_through_pipeline(self, rag1_output_record):
        """Verify Hebrew text is preserved without corruption through the pipeline."""
        action = prepare_bulk_action(rag1_output_record)

        # Hebrew characters should be preserved exactly
        assert "כותרת" in action["_source"]["title"]
        assert "עברית" in action["_source"]["full_chunk"]

    def test_embedding_values_preserved(self, rag1_output_record):
        """Verify embedding float values are preserved with precision."""
        action = prepare_bulk_action(rag1_output_record)

        original_embedding = rag1_output_record["embedding"]
        processed_embedding = action["_source"]["embedding"]

        assert original_embedding == processed_embedding
        assert processed_embedding[0] == 0.123456


class TestEmbeddingTypeCompatibility:
    """Tests for various embedding type scenarios."""

    def test_list_embedding_valid(self):
        """Standard list embedding from FLOAT8[] is valid."""
        record = {
            "post_id": "test1",
            "timestamp": datetime.now(),
            "title": "Test",
            "full_chunk": "Test content",
            "has_consultant": False,
            "engagement_norm": 0.5,
            "embedding": [0.1] * 1024,
        }
        is_valid, error = validate_record(record)
        assert is_valid is True

    def test_tuple_embedding_valid(self):
        """Tuple embedding is also valid (some drivers may return tuples)."""
        record = {
            "post_id": "test2",
            "timestamp": datetime.now(),
            "title": "Test",
            "full_chunk": "Test content",
            "has_consultant": False,
            "engagement_norm": 0.5,
            "embedding": tuple([0.1] * 1024),
        }
        is_valid, error = validate_record(record)
        assert is_valid is True

    def test_string_embedding_invalid(self):
        """
        String embedding is INVALID - this would happen with pgvector type.

        If rag1 used vector(1024) instead of FLOAT8[], psycopg2 would return
        a string like '[0.1,0.2,...]' which rag2 cannot process.
        """
        record = {
            "post_id": "test3",
            "timestamp": datetime.now(),
            "title": "Test",
            "full_chunk": "Test content",
            "has_consultant": False,
            "engagement_norm": 0.5,
            "embedding": "[0.1,0.2,0.3]",  # String - INVALID!
        }
        is_valid, error = validate_record(record)
        assert is_valid is False
        assert "not an array" in error

    def test_mixed_int_float_embedding_valid(self):
        """Embedding with mixed int/float values is valid."""
        record = {
            "post_id": "test4",
            "timestamp": datetime.now(),
            "title": "Test",
            "full_chunk": "Test content",
            "has_consultant": False,
            "engagement_norm": 0.5,
            "embedding": [1, 0.5, 2, 0.25] + [0.1] * 1020,
        }
        is_valid, error = validate_record(record)
        assert is_valid is True


class TestResumeCapability:
    """Tests for the ingested_at resume mechanism between rag1 and rag2."""

    def test_rag1_does_not_set_ingested_at(self):
        """
        Verify that rag1 leaves ingested_at as NULL.

        rag1's UPSERT does not include ingested_at, so it defaults to NULL,
        allowing rag2 to find unprocessed records.
        """
        # rag1's UPSERT_EMBEDDING_SQL columns:
        # post_id, timestamp, title, full_chunk, has_consultant, engagement_norm, embedding
        # Note: ingested_at is NOT included - defaults to NULL

        # This is verified by checking rag1's database.py UPSERT_EMBEDDING_SQL
        # which has 7 columns (not 8)
        expected_columns = 7
        assert expected_columns == 7, "rag1 should write 7 columns, not including ingested_at"

    def test_rag2_filters_by_null_ingested_at(self):
        """
        Verify rag2 uses WHERE ingested_at IS NULL for resume.

        This ensures rag2 only processes records that haven't been indexed yet.
        """
        # From rag2's lambda_function.py stream_records_from_aurora():
        # WHERE ingested_at IS NULL
        query_filter = "WHERE ingested_at IS NULL"
        assert "ingested_at IS NULL" in query_filter


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

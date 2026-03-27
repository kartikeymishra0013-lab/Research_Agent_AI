"""
Tests for extraction layer — schema extractor and relationship extractor.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

from src.extraction.schema_extractor import SchemaExtractor
from src.extraction.relationship_extractor import RelationshipExtractor


class TestSchemaExtractor:
    def test_extract_returns_dict(self, mock_openai_client, sample_document):
        extractor = SchemaExtractor(
            client=mock_openai_client,
            schema_type="research_paper",
        )
        result = extractor.extract(sample_document.content, doc_metadata=sample_document.metadata)

        assert isinstance(result, dict)
        assert "title" in result

    def test_extract_empty_text_returns_empty(self, mock_openai_client):
        extractor = SchemaExtractor(client=mock_openai_client, schema_type="default")
        result = extractor.extract("")
        assert result == {}

    def test_extract_uses_correct_schema_type(self, mock_openai_client):
        extractor = SchemaExtractor(client=mock_openai_client, schema_type="research_paper")
        assert extractor.schema_type == "research_paper"
        # Schema should have research paper-specific fields
        assert "abstract" in extractor.schema["fields"]
        assert "methodology" in extractor.schema["fields"]

    def test_extract_patent_schema(self, mock_openai_client):
        # Mock patent extraction response
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = json.dumps({
            "patent_title": "Method and System for Neural Processing",
            "patent_number": "US10123456B2",
            "inventors": ["John Doe"],
            "assignee": "TechCorp Inc.",
        })
        extractor = SchemaExtractor(client=mock_openai_client, schema_type="patent")
        result = extractor.extract("This patent describes a neural processing method...")
        assert "patent_title" in result or "title" in result

    def test_invalid_json_response_handled(self, mock_openai_client, sample_document):
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = "not json {"
        extractor = SchemaExtractor(client=mock_openai_client)
        result = extractor.extract(sample_document.content)
        assert "raw_response" in result


class TestRelationshipExtractor:
    def test_extract_returns_entities_and_relationships(self, mock_openai_client, sample_document):
        # Mock relationship extraction response
        mock_response = {
            "entities": [
                {"id": "transformer_model", "label": "Transformer", "type": "Technology", "properties": {}},
                {"id": "vaswani_et_al", "label": "Vaswani et al.", "type": "Person", "properties": {}},
            ],
            "relationships": [
                {"source": "vaswani_et_al", "target": "transformer_model", "type": "DEVELOPED_BY", "properties": {}}
            ]
        }
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = json.dumps(mock_response)

        extractor = RelationshipExtractor(client=mock_openai_client)
        result = extractor.extract(sample_document.content, doc_id="test_doc_001")

        assert "entities" in result
        assert "relationships" in result
        assert len(result["entities"]) == 2
        assert len(result["relationships"]) == 1

    def test_extract_empty_text_returns_empty(self, mock_openai_client):
        extractor = RelationshipExtractor(client=mock_openai_client)
        result = extractor.extract("")
        assert result == {"entities": [], "relationships": []}

    def test_doc_id_injected_into_properties(self, mock_openai_client):
        mock_response = {
            "entities": [{"id": "ent1", "label": "Entity 1", "type": "Concept", "properties": {}}],
            "relationships": []
        }
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = json.dumps(mock_response)

        extractor = RelationshipExtractor(client=mock_openai_client)
        result = extractor.extract("Some text", doc_id="doc_abc123")

        assert result["entities"][0]["properties"]["doc_id"] == "doc_abc123"

    def test_invalid_json_response_returns_empty(self, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = "not json"
        extractor = RelationshipExtractor(client=mock_openai_client)
        result = extractor.extract("Some text")
        assert result == {"entities": [], "relationships": []}

import unittest
import json
import sys
from unittest.mock import MagicMock, patch

# Mock google.genai before importing classifier
mock_genai = MagicMock()
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = mock_genai

from src.classifier import Classifier, Keyword, format_keywords, format_keywords_by_category

class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.mock_keywords = [
            Keyword(id=1, name="Root", level=0, parent_id=None, full_path="Root", indented_name="Root"),
            Keyword(id=2, name="Leaf", level=1, parent_id=1, full_path="Root > Leaf", indented_name="  Leaf"),
            Keyword(id=3, name="Category2", level=0, parent_id=None, full_path="Category2", indented_name="Category2"),
            Keyword(id=4, name="SubItem", level=1, parent_id=3, full_path="Category2 > SubItem", indented_name="  SubItem")
        ]

    @patch('src.classifier.OpenAI')
    def test_openai_classify(self, mock_openai):
        # Setup mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # To make side_effect work for two calls (classify + suggest):
        mock_response_1 = MagicMock()
        mock_response_1.choices[0].message.content = "2"
        mock_response_2 = MagicMock()
        mock_response_2.choices[0].message.content = "NONE"
        mock_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]

        classifier = Classifier(provider="openai", api_key="fake")

        matched, new_kws, raw = classifier.classify("Some text about Leaf", self.mock_keywords)
        
        self.assertEqual(matched, ["2"])
        self.assertEqual(new_kws, [])

    def test_format_keywords(self):
        formatted = format_keywords(self.mock_keywords)
        self.assertIn("- Root (ID: 1)", formatted)
        self.assertIn("  - Leaf (ID: 2)", formatted)

    def test_format_keywords_by_category(self):
        formatted = format_keywords_by_category(self.mock_keywords)
        self.assertIn("Category Root, id: 1", formatted)
        self.assertIn("  - Leaf (id: 2)", formatted)
        self.assertIn("Category Category2, id: 3", formatted)
        self.assertIn("  - SubItem (id: 4)", formatted)

    @patch('src.classifier.OpenAI')
    def test_classify_match_keywords(self, mock_openai):
        """Test MATCH_KEYWORDS prompt path with JSON response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        llm_json = json.dumps([
            {"category": "Root", "keyword": "Leaf", "suggested": False, "category_id": 1, "keyword_id": 2},
            {"category": "other", "keyword": "new_kw", "suggested": True, "category_id": -1, "keyword_id": -1}
        ])
        mock_response.choices[0].message.content = llm_json
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        classifier = Classifier(
            provider="openai", api_key="fake",
            prompt_name="MATCH_KEYWORDS"
        )

        matched_ids, suggested, raw = classifier.classify(
            "Some text", self.mock_keywords,
            {"language": "Hebrew", "translation": "Some translation"}
        )
        self.assertEqual(matched_ids, ["2"])
        self.assertEqual(suggested, ["new_kw"])

if __name__ == '__main__':
    unittest.main()

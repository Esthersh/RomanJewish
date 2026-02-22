import unittest
from unittest.mock import MagicMock, patch
from src.classifier import Classifier, Keyword

class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.mock_keywords = [
            Keyword(id=1, name="Root", level=0, parent_id=None, full_path="Root", indented_name="Root"),
            Keyword(id=2, name="Leaf", level=1, parent_id=1, full_path="Root > Leaf", indented_name="  Leaf")
        ]

    @patch('src.classifier.OpenAI')
    def test_openai_classify(self, mock_openai):
        # Setup mock response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "2"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        classifier = Classifier(provider="openai", api_key="fake")
        
        # Test step 1 (classification) being 2, and step 2 (suggestion) being NONE (by reusing mock behavior or setting up side_effect)
        # To make side_effect work for two calls:
        mock_response_1 = MagicMock()
        mock_response_1.choices[0].message.content = "2"
        mock_response_2 = MagicMock()
        mock_response_2.choices[0].message.content = "NONE"
        mock_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]

        matched, new_kws = classifier.classify("Some text about Leaf", self.mock_keywords)
        
        self.assertEqual(matched, ["2"])
        self.assertEqual(new_kws, [])

    def test_format_keywords(self):
        classifier = Classifier(provider="openai", api_key="fake")
        formatted = format_keywords(self.mock_keywords)
        self.assertIn("- Root (ID: 1)", formatted)
        self.assertIn("  - Leaf (ID: 2)", formatted)

if __name__ == '__main__':
    unittest.main()

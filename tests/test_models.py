import unittest
from src.models import MatchKeywordEntry, validate_match_keywords_response
from pydantic import ValidationError


class TestMatchKeywordEntry(unittest.TestCase):
    def test_valid_entry(self):
        entry = MatchKeywordEntry(
            category="Property",
            keyword="ownership",
            suggested=False,
            category_id=7,
            keyword_id=42
        )
        self.assertEqual(entry.category, "Property")
        self.assertEqual(entry.keyword_id, 42)
        self.assertFalse(entry.suggested)

    def test_suggested_entry(self):
        entry = MatchKeywordEntry(
            category="other",
            keyword="new concept",
            suggested=True,
            category_id=-1,
            keyword_id=-1
        )
        self.assertTrue(entry.suggested)
        self.assertEqual(entry.category_id, -1)

    def test_invalid_entry_missing_field(self):
        with self.assertRaises(ValidationError):
            MatchKeywordEntry(
                category="Property",
                keyword="ownership",
                # missing suggested, category_id, keyword_id
            )

    def test_invalid_entry_wrong_type(self):
        with self.assertRaises(ValidationError):
            MatchKeywordEntry(
                category="Property",
                keyword="ownership",
                suggested="not_a_bool",
                category_id="not_an_int",
                keyword_id=42
            )

    def test_validate_match_keywords_response(self):
        raw = [
            {"category": "A", "keyword": "kw1", "suggested": False, "category_id": 1, "keyword_id": 10},
            {"category": "B", "keyword": "kw2", "suggested": True, "category_id": -1, "keyword_id": -1},
        ]
        entries = validate_match_keywords_response(raw)
        self.assertEqual(len(entries), 2)
        self.assertFalse(entries[0].suggested)
        self.assertTrue(entries[1].suggested)

    def test_validate_match_keywords_response_invalid(self):
        raw = [{"category": "A"}]  # missing required fields
        with self.assertRaises(ValidationError):
            validate_match_keywords_response(raw)


if __name__ == '__main__':
    unittest.main()

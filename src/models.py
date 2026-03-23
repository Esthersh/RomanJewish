from pydantic import BaseModel
from typing import List


class MatchKeywordEntry(BaseModel):
    """Validates a single keyword match from the MATCH_KEYWORDS prompt response."""
    category: str
    keyword: str
    suggested: bool
    category_id: int
    keyword_id: int


def validate_match_keywords_response(raw_json: list) -> List[MatchKeywordEntry]:
    """
    Validate and parse the raw JSON list from the MATCH_KEYWORDS LLM response.
    Returns a list of validated MatchKeywordEntry objects.
    Raises ValidationError if any entry is malformed.
    """
    return [MatchKeywordEntry(**entry) for entry in raw_json]


class MatchFieldEntry(BaseModel):
    """Validates a single field match from the FIELDS prompt response."""
    field: str
    field_id: int


def validate_match_fields_response(raw_json: list) -> List[MatchFieldEntry]:
    """
    Validate and parse the raw JSON list from the FIELDS LLM response.
    Returns a list of validated MatchFieldEntry objects.
    Raises ValidationError if any entry is malformed.
    """
    return [MatchFieldEntry(**entry) for entry in raw_json]


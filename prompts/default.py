CLASSIFICATION_PROMPT = """You are an expert in legal traditions endured under the rule of the Roman Empire.

Classify the following text using the provided keyword hierarchy.
Return ONLY the specific Level 1 keyword IDs that match the text best.
If the text fits a Level 0 category but no specific Level 1 keyword applies, do not select any.
Return the IDs as a list of tuples (id, word).

Context:
Group: {group}
Name: {name}

Keywords:
{hierarchy_str}

Text:
{text}

Matched Keyword:
"""

SUGGESTION_PROMPT = """You are an expert in legal traditions endured under the rule of the Roman Empire.
Analyze the following text.

Does the current keyword list cover the legal topics discussed in the text?
If there are important legal concepts in the text MISSING from the list, suggest new short keywords (1-3 words) in English.
Do NOT suggest keywords that are synonyms of existing ones.
Return ONLY the new keywords as a comma-separated list. If none, return "NONE".

Context:
Group: {group}
Name: {name}

Current Keywords:
{hierarchy_str}

Text:
{text}

New Keywords:"""

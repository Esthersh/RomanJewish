KEYWORDS = """## Context
Match the provided legal source to the most relevant keywords covering ideas, terms, and objects from the hierarchy provided below. The keyword hierarchy is organized by overarching categories.

## Instructions
1. Analyze the provided {language} legal text.
2. Review the provided Keyword hierarchy.
3. Identify and return the keywords from the hierarchy that best match the ideas, terms, and objects in the text.
4. If the hierarchy does not adequately cover the source text, you may suggest alternative keywords. 
    * All suggested keywords MUST be in English.
    * Suggest up to 3 additional most relevant keywords.
    * Assign any suggested keyword to the existing category that matches it best. If no existing category fits, use "other" for the category field.
    * For suggested keywords, set the keyword_id to -1. Set the category_id to the ID of the matched category, or -1 if the category is "other".

## Keyword Hierarchy 
{hierarchy}

## Output Format
Return ONLY a valid JSON array of objects. Do not include markdown formatting like ```json, preamble, or conversational text. Use the exact data types and structure shown in this example:

[
  {{
    "category": "Name of an existing category",
    "keyword": "An exact match from the hierarchy",
    "suggested": false, 
    "category_id": 123,
    "keyword_id": 456
  }},
  {{
    "category": "other",
    "keyword": "A brand new suggested keyword",
    "suggested": true, 
    "category_id": -1,
    "keyword_id": -1
  }}
]

## Input Data
Source Text, {source_name} ({language}): 
{text}
"""

KEYWORDS_W_EN = """## Context
Match the provided legal source to the most relevant keywords covering ideas, terms, and objects from the hierarchy provided below. The keyword hierarchy is organized by overarching categories.

## Instructions
1. Analyze the provided {language} legal text and its English translation.
2. Review the provided Keyword hierarchy.
3. Identify and return the keywords from the hierarchy that best match the ideas, terms, and objects in the text.
4. If the hierarchy does not adequately cover the source text, you may suggest alternative keywords. 
    * All suggested keywords MUST be in English.
    * Suggest up to 3 additional most relevant keywords.
    * Assign any suggested keyword to the existing category that matches it best. If no existing category fits, use "other" for the category field.
    * For suggested keywords, set the keyword_id to -1. Set the category_id to the ID of the matched category, or -1 if the category is "other".

## Keyword Hierarchy 
{hierarchy}

## Output Format
Return ONLY a valid JSON array of objects. Do not include markdown formatting like ```json, preamble, or conversational text. Use the exact data types and structure shown in this example:

[
  {{
    "category": "Name of an existing category",
    "keyword": "An exact match from the hierarchy",
    "suggested": false, 
    "category_id": 123,
    "keyword_id": 456
  }},
  {{
    "category": "other",
    "keyword": "A brand new suggested keyword",
    "suggested": true, 
    "category_id": -1,
    "keyword_id": -1
  }}
]

## Input Data
Source Text, {source_name} ({language}): 
{text}

English Translation: 
{translation}
"""

FIELDS = """## Context
Match the provided legal source to the most relevant judicial fields from the hierarchy provided below. 

## Instructions
1. Analyze the provided {language} legal text.
2. Review the provided judicial field hierarchy.
3. Identify and return the fields from the hierarchy that best match the text.
4. Select the lowest level possible that is relevant; otherwise, choose a relevant field from the level above.
5. If none of the fields in the hierarchy adequately cover the source, return an empty list. 

## Judicial Topic Hierarchy 
{hierarchy}

## Output Format
Return ONLY a valid JSON array of objects. Do not include markdown formatting like ```json, preamble, or conversational text. Use the exact data types and structure shown in this example:

[
  {{
    "field": level_0 > level_1 > level_2,
    "field_id": 123,
  }},
  {{
    "field": level_0,
    "field_id": 38,
  }}
]

## Input Data
Source Text, {source_name} ({language}): 
{text}
"""

FIELDS_W_EN = """## Context
Match the provided legal source to the most relevant judicial fields from the hierarchy provided below. 

## Instructions
1. Analyze the provided {language} legal text and its English translation.
2. Review the provided judicial field hierarchy.
3. Identify and return the fields from the hierarchy that best match the text.
4. Select the lowest level possible that is relevant; otherwise, choose a relevant field from the level above.
5. If none of the fields in the hierarchy adequately cover the source, return an empty list. 

## Judicial Topic Hierarchy 
{hierarchy}

## Output Format
Return ONLY a valid JSON array of objects. Do not include markdown formatting like ```json, preamble, or conversational text. Use the exact data types and structure shown in this example:

[
  {{
    "field": level_0 > level_1 > level_2,
    "field_id": 123,
  }},
  {{
    "field": level_0,
    "field_id": 38,
  }}
]

## Input Data
Source Text, {source_name} ({language}): 
{text}

English Translation: 
{translation}
"""

INDEX = """## Context
We are building an index for our corpus of legal sources.

## Instructions
1. Analyze the provided source.
2. Select the most appropriate terms or short phrases within the source to include in the index.
2. Return the dictionary form of the selected terms or phrases.

## Output Format
Return ONLY a valid JSON array of strings. Do not include markdown formatting, headers, or conversational text.

Example format:
["term1", "term2", "term3"]

## Input Data
Source Text, {source_name} ({language}): 
{text}
"""

INDEX_W_EN = """## Context
We are building an index for our corpus of legal sources.

## Instructions
1. Analyze the provided source and its English translation.
2. Select the most appropriate terms or short phrases within the source to include in the index.
3. Return the dictionary form of the selected terms or phrases.

## Output Format
Return ONLY a valid JSON array of strings. Do not include markdown formatting, headers, or conversational text.

Example format:
["term1", "term2", "term3"]

## Input Data
Source Text, {source_name} ({language}): 
{text}

English Translation: 
{translation}
"""

INDEX_V1 = """## Context
We are building an index for our corpus of legal sources.

## Instructions
1. Analyze the provided source.
2. Select terms or short phrases that are distinctive to the specific legal issues discussed in the text. 
3. Exclude very common, high-frequency words and general vocabulary (for instance: "son", "daughter", "man", "day", "house") 
4. Return the dictionary form (lemma) of the selected terms or phrases.


## Output Format
Return ONLY a valid JSON array of strings. Do not include markdown formatting, headers, or conversational text.

Example format:
["term1", "term2", "term3"]

## Input Data
Source Text, {source_name} ({language}): 
{text}
"""

INDEX_W_EN_V1 = """## Context
We are building an index for our corpus of legal sources.

## Instructions
1. Analyze the provided source and its English translation.
2. Select terms or short phrases that are distinctive to the specific legal issues discussed in the text. 
3. Exclude very common, high-frequency words and general vocabulary (for instance: "son", "daughter", "man", "day", "house") 
4. Return the dictionary form (lemma) of the selected terms or phrases.


## Output Format
Return ONLY a valid JSON array of strings. Do not include markdown formatting, headers, or conversational text.

Example format:
["term1", "term2", "term3"]

## Input Data
Source Text, {source_name} ({language}): 
{text}

English Translation: 
{translation}
"""
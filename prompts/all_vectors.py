KEYWORDS = """## Objective
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

KEYWORDS_CONTEXT = """## Objective
Match the provided legal source to the most relevant keywords covering ideas, terms, and objects from the hierarchy provided below. The keyword hierarchy is organized by overarching categories.

## Instructions
1. Analyze the provided {language} Target Text. 
2. Read the "Broader Context" provided. Use this context strictly to clarify ambiguities, identify underlying themes, or understand references within the Target Text.
3. Review the provided Keyword hierarchy.
4. Identify and return the keywords from the hierarchy that best match the ideas, terms, and objects in the text.
5. If the hierarchy does not adequately cover the source text, you may suggest alternative keywords. 
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
Broader Context (e.g., Full Chapter):
{broader_context}

Target Source Text, {source_name} ({language}): 
{text}
"""

KEYWORDS_W_EN = """## Objective
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

KEYWORDS_W_EN_CONTEXT = """## Objective
Match the provided legal source to the most relevant keywords covering ideas, terms, and objects from the hierarchy provided below. The keyword hierarchy is organized by overarching categories.

## Instructions
1. Analyze the provided {language} Target Text and its English translation. 
2. Read the "Broader Context" provided. Use this context strictly to clarify ambiguities, identify underlying themes, or understand references within the Target Text.
3. Review the provided Keyword hierarchy.
4. Identify and return the keywords from the hierarchy that best match the ideas, terms, and objects in the text.
5. If the hierarchy does not adequately cover the source text, you may suggest alternative keywords. 
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
Broader Context (e.g., Full Chapter):
{broader_context}

Target Source Text, {source_name} ({language}): 
{text}

English Translation: 
{translation}
"""

# ###################################
FIELDS = """## Objective
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
    "field_id": 123
  }},
  {{
    "field": level_0,
    "field_id": 38
  }}
]

## Input Data
Source Text, {source_name} ({language}): 
{text}
"""

FIELDS_CONTEXT = """## Objective
Match the provided legal source to the most relevant judicial fields from the hierarchy provided below. 

## Instructions
1. Analyze the provided {language} legal text.
2.Read the "Broader Context" provided. Use this context strictly to clarify ambiguities, identify underlying themes, or understand references within the Target Text.
3. Review the provided judicial field hierarchy.
4. Identify and return the fields from the hierarchy that best match the text.
5. Select the lowest level possible that is relevant; otherwise, choose a relevant field from the level above.
6. If none of the fields in the hierarchy adequately cover the source, return an empty list. 

## Judicial Topic Hierarchy 
Courts and Procedure (ID 4)
Courts and Procedure > Appeal (ID 20)
Courts and Procedure > Courts (ID 16)
Courts and Procedure > Courts > Arbitration (ID 100)
Courts and Procedure > Courts > Choice of Judges (ID 99)
Courts and Procedure > Evidence (ID 17)
Courts and Procedure > Evidence > Archiving (ID 109)
Courts and Procedure > Evidence > Documentary Evidence (ID 102)
Courts and Procedure > Evidence > Witnesses (ID 101)
Courts and Procedure > Execution (ID 19)
Courts and Procedure > Oaths (ID 18)
Courts and Procedure > Procedure (ID 121)
Courts and Procedure > Procedure > Confession (ID 122)
Obligations (ID 3)
Obligations > Consensual Contract (ID 14)
Obligations > Consensual Contract > Deposit (ID 79)
Obligations > Consensual Contract > Deposit > Responsibility of depositee (ID 90)
Obligations > Consensual Contract > Letting and Hiring (ID 78)
Obligations > Consensual Contract > Letting and Hiring > Emphyteusis (ID 88)
Obligations > Consensual Contract > Letting and Hiring > Labour (ID 85)
Obligations > Consensual Contract > Letting and Hiring > Land Tenancy (ID 86)
Obligations > Consensual Contract > Letting and Hiring > Misthosis (ID 89)
Obligations > Consensual Contract > Loan and Pledge (ID 74)
Obligations > Consensual Contract > Loan and Pledge > Interest (ID 80)
Obligations > Consensual Contract > Loan and Pledge > Real security (ID 81)
Obligations > Consensual Contract > Loan and Pledge > Transfer of Debt (ID 82)
Obligations > Consensual Contract > Mandate (ID 77)
Obligations > Consensual Contract > Partnership (ID 76)
Obligations > Consensual Contract > Sale (ID 75)
Obligations > Consensual Contract > Sale > Fraud (ID 84)
Obligations > Consensual Contract > Sale > Warrenty (ID 83)
Obligations > Delicts (ID 15)
Obligations > Delicts > Damage (ID 93)
Obligations > Delicts > Damage > Indirect cause (ID 98)
Obligations > Delicts > Damage > wild Animals (ID 97)
Obligations > Delicts > Theft (ID 91)
Obligations > Delicts > Theft > In deposit (ID 96)
Obligations > Delicts > Theft > Intention (ID 95)
Obligations > Delicts > Theft > Return of Object (ID 94)
Obligations > Delicts > Wild Animals (ID 92)
Obligations > Verbal Contracts (ID 107)
Obligations > Verbal Contracts > Stipulation (ID 108)
Property (ID 2)
Property > Acquisition (ID 10)
Property > Acquisition > Conveyance (ID 55)
Property > Acquisition > Conveyance > Classification of Things (ID 60)
Property > Acquisition > Conveyance > Delivery (ID 62)
Property > Acquisition > Conveyance > Formal act (ID 61)
Property > Acquisition > Conveyance > Payment (ID 63)
Property > Acquisition > Conveyance > Usucapio (ID 64)
Property > Acquisition > Occupation (ID 56)
Property > Acquisition > Occupation > Abandoned Property (ID 68)
Property > Acquisition > Occupation > Accession (ID 69)
Property > Acquisition > Occupation > From Natural state (ID 65)
Property > Acquisition > Occupation > Specification (ID 67)
Property > Acquisition > Occupation > Superficies (ID 66)
Property > Acquisition > Usufruct (ID 57)
Property > Neighbors (ID 13)
Property > Possession (ID 11)
Property > Servitudes (ID 12)
Property > Servitudes > Personal (ID 59)
Property > Servitudes > Personal > Usufruct (ID 73)
Property > Servitudes > Praedial (ID 58)
Property > Servitudes > Praedial > Rights of Light (ID 72)
Property > Servitudes > Praedial > Rights of Water (ID 71)
Property > Servitudes > Praedial > Rights of Way (ID 70)
Public Law (ID 110)
Public Law > Administrative Law (ID 112)
Public Law > Administrative Law > Archive (ID 114)
Public Law > Administrative Law > Judiciary (ID 113)
Public Law > Administrative Law > Municipal Government  (ID 115)
Public Law > Administrative Law > Provincial Government  (ID 123)
Public Law > Taxation (ID 111)
Status (ID 1)
Status > Citizenship (ID 7)
Status > Citizenship > Captivity (ID 41)
Status > Citizenship > Freed Persons (ID 105)
Status > Citizenship > Naturalization (ID 40)
Status > Family (ID 5)
Status > Family > Divorce (ID 22)
Status > Family > Marital Arrangements (ID 23)
Status > Family > Marital Arrangements > Dowry (ID 24)
Status > Family > Marital Arrangements > Property of Women (ID 27)
Status > Family > Marital Arrangements > Provision for Wife (ID 25)
Status > Family > Marital Arrangements > Provisions for Children (ID 26)
Status > Family > Marriage (ID 21)
Status > Family > Marriage > Acts of Marriage (ID 117)
Status > Family > Marriage > Legitimate Children (ID 118)
Status > Inheritance (ID 6)
Status > Inheritance > Debts and Obligations (ID 119)
Status > Inheritance > Intestacy (ID 28)
Status > Inheritance > Intestacy > Classes of Heirs (ID 30)
Status > Inheritance > Intestacy > Father Inheritance Rights  (ID 103)
Status > Inheritance > Intestacy > Firstborn Inheritance Rights (ID 104)
Status > Inheritance > Intestacy > Women Inheritance Rights (ID 31)
Status > Inheritance > Testaments (ID 29)
Status > Inheritance > Testaments > Acceptance of will (ID 35)
Status > Inheritance > Testaments > Capacity (ID 34)
Status > Inheritance > Testaments > Causa Mortis (ID 32)
Status > Inheritance > Testaments > Deeds of Gift as Wills (ID 37)
Status > Inheritance > Testaments > Disinheritance (ID 33)
Status > Inheritance > Testaments > Forms of Will (ID 36)
Status > Inheritance > Testaments > Legacies (ID 38)
Status > Inheritance > Testaments > Trust (ID 39)
Status > Legal Capacity (ID 9)
Status > Legal Capacity > Adoption (ID 47)
Status > Legal Capacity > Guardianship (ID 48)
Status > Legal Capacity > Guardianship > Guardianship over Minors (ID 53)
Status > Legal Capacity > Guardianship > Guardianship over Women (ID 54)
Status > Legal Capacity > Infamy (ID 51)
Status > Legal Capacity > Legal Capacity of Women (ID 120)
Status > Legal Capacity > Mental Disability (ID 50)
Status > Legal Capacity > Minors (ID 49)
Status > Legal Capacity > Paternal Power (ID 46)
Status > Legal Capacity > Paternal Power > Emanciaption (ID 52)
Status > Slavery (ID 8)
Status > Slavery > Manumission (ID 43)
Status > Slavery > Manumission > Paramone (ID 45)
Status > Slavery > Manumission > Pseudo Manumissions (ID 44)
Status > Slavery > Obligations of Slaves (ID 42)
Status > Slavery > Patronage/Clientela (ID 106)


## Output Format
Return ONLY a valid JSON array of objects. Do not include markdown formatting like ```json, preamble, or conversational text. Use the exact data types and structure shown in this example:

[
  {{
    "field": level_0 > level_1 > level_2,
    "field_id": 123
  }},
  {{
    "field": level_0,
    "field_id": 38
  }}
]

## Input Data
Broader Context (e.g., Full Chapter):
{broader_context}

Target Source Text, {source_name} ({language}): 
{text}
"""

FIELDS_W_EN = """## Objective
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
    "field_id": 123
  }},
  {{
    "field": level_0,
    "field_id": 38
  }}
]

## Input Data
Source Text, {source_name} ({language}): 
{text}

English Translation: 
{translation}
"""

FIELDS_W_EN_CONTEXT = """## Objective
Match the provided legal source to the most relevant judicial fields from the hierarchy provided below. 

## Instructions
1. Analyze the provided {language} legal text and its English translation.
2.Read the "Broader Context" provided. Use this context strictly to clarify ambiguities, identify underlying themes, or understand references within the Target Text.
3. Review the provided judicial field hierarchy.
4. Identify and return the fields from the hierarchy that best match the text.
5. Select the lowest level possible that is relevant; otherwise, choose a relevant field from the level above.
6. If none of the fields in the hierarchy adequately cover the source, return an empty list. 

## Judicial Topic Hierarchy 
Courts and Procedure (ID 4)
Courts and Procedure > Appeal (ID 20)
Courts and Procedure > Courts (ID 16)
Courts and Procedure > Courts > Arbitration (ID 100)
Courts and Procedure > Courts > Choice of Judges (ID 99)
Courts and Procedure > Evidence (ID 17)
Courts and Procedure > Evidence > Archiving (ID 109)
Courts and Procedure > Evidence > Documentary Evidence (ID 102)
Courts and Procedure > Evidence > Witnesses (ID 101)
Courts and Procedure > Execution (ID 19)
Courts and Procedure > Oaths (ID 18)
Courts and Procedure > Procedure (ID 121)
Courts and Procedure > Procedure > Confession (ID 122)
Obligations (ID 3)
Obligations > Consensual Contract (ID 14)
Obligations > Consensual Contract > Deposit (ID 79)
Obligations > Consensual Contract > Deposit > Responsibility of depositee (ID 90)
Obligations > Consensual Contract > Letting and Hiring (ID 78)
Obligations > Consensual Contract > Letting and Hiring > Emphyteusis (ID 88)
Obligations > Consensual Contract > Letting and Hiring > Labour (ID 85)
Obligations > Consensual Contract > Letting and Hiring > Land Tenancy (ID 86)
Obligations > Consensual Contract > Letting and Hiring > Misthosis (ID 89)
Obligations > Consensual Contract > Loan and Pledge (ID 74)
Obligations > Consensual Contract > Loan and Pledge > Interest (ID 80)
Obligations > Consensual Contract > Loan and Pledge > Real security (ID 81)
Obligations > Consensual Contract > Loan and Pledge > Transfer of Debt (ID 82)
Obligations > Consensual Contract > Mandate (ID 77)
Obligations > Consensual Contract > Partnership (ID 76)
Obligations > Consensual Contract > Sale (ID 75)
Obligations > Consensual Contract > Sale > Fraud (ID 84)
Obligations > Consensual Contract > Sale > Warrenty (ID 83)
Obligations > Delicts (ID 15)
Obligations > Delicts > Damage (ID 93)
Obligations > Delicts > Damage > Indirect cause (ID 98)
Obligations > Delicts > Damage > wild Animals (ID 97)
Obligations > Delicts > Theft (ID 91)
Obligations > Delicts > Theft > In deposit (ID 96)
Obligations > Delicts > Theft > Intention (ID 95)
Obligations > Delicts > Theft > Return of Object (ID 94)
Obligations > Delicts > Wild Animals (ID 92)
Obligations > Verbal Contracts (ID 107)
Obligations > Verbal Contracts > Stipulation (ID 108)
Property (ID 2)
Property > Acquisition (ID 10)
Property > Acquisition > Conveyance (ID 55)
Property > Acquisition > Conveyance > Classification of Things (ID 60)
Property > Acquisition > Conveyance > Delivery (ID 62)
Property > Acquisition > Conveyance > Formal act (ID 61)
Property > Acquisition > Conveyance > Payment (ID 63)
Property > Acquisition > Conveyance > Usucapio (ID 64)
Property > Acquisition > Occupation (ID 56)
Property > Acquisition > Occupation > Abandoned Property (ID 68)
Property > Acquisition > Occupation > Accession (ID 69)
Property > Acquisition > Occupation > From Natural state (ID 65)
Property > Acquisition > Occupation > Specification (ID 67)
Property > Acquisition > Occupation > Superficies (ID 66)
Property > Acquisition > Usufruct (ID 57)
Property > Neighbors (ID 13)
Property > Possession (ID 11)
Property > Servitudes (ID 12)
Property > Servitudes > Personal (ID 59)
Property > Servitudes > Personal > Usufruct (ID 73)
Property > Servitudes > Praedial (ID 58)
Property > Servitudes > Praedial > Rights of Light (ID 72)
Property > Servitudes > Praedial > Rights of Water (ID 71)
Property > Servitudes > Praedial > Rights of Way (ID 70)
Public Law (ID 110)
Public Law > Administrative Law (ID 112)
Public Law > Administrative Law > Archive (ID 114)
Public Law > Administrative Law > Judiciary (ID 113)
Public Law > Administrative Law > Municipal Government  (ID 115)
Public Law > Administrative Law > Provincial Government  (ID 123)
Public Law > Taxation (ID 111)
Status (ID 1)
Status > Citizenship (ID 7)
Status > Citizenship > Captivity (ID 41)
Status > Citizenship > Freed Persons (ID 105)
Status > Citizenship > Naturalization (ID 40)
Status > Family (ID 5)
Status > Family > Divorce (ID 22)
Status > Family > Marital Arrangements (ID 23)
Status > Family > Marital Arrangements > Dowry (ID 24)
Status > Family > Marital Arrangements > Property of Women (ID 27)
Status > Family > Marital Arrangements > Provision for Wife (ID 25)
Status > Family > Marital Arrangements > Provisions for Children (ID 26)
Status > Family > Marriage (ID 21)
Status > Family > Marriage > Acts of Marriage (ID 117)
Status > Family > Marriage > Legitimate Children (ID 118)
Status > Inheritance (ID 6)
Status > Inheritance > Debts and Obligations (ID 119)
Status > Inheritance > Intestacy (ID 28)
Status > Inheritance > Intestacy > Classes of Heirs (ID 30)
Status > Inheritance > Intestacy > Father Inheritance Rights  (ID 103)
Status > Inheritance > Intestacy > Firstborn Inheritance Rights (ID 104)
Status > Inheritance > Intestacy > Women Inheritance Rights (ID 31)
Status > Inheritance > Testaments (ID 29)
Status > Inheritance > Testaments > Acceptance of will (ID 35)
Status > Inheritance > Testaments > Capacity (ID 34)
Status > Inheritance > Testaments > Causa Mortis (ID 32)
Status > Inheritance > Testaments > Deeds of Gift as Wills (ID 37)
Status > Inheritance > Testaments > Disinheritance (ID 33)
Status > Inheritance > Testaments > Forms of Will (ID 36)
Status > Inheritance > Testaments > Legacies (ID 38)
Status > Inheritance > Testaments > Trust (ID 39)
Status > Legal Capacity (ID 9)
Status > Legal Capacity > Adoption (ID 47)
Status > Legal Capacity > Guardianship (ID 48)
Status > Legal Capacity > Guardianship > Guardianship over Minors (ID 53)
Status > Legal Capacity > Guardianship > Guardianship over Women (ID 54)
Status > Legal Capacity > Infamy (ID 51)
Status > Legal Capacity > Legal Capacity of Women (ID 120)
Status > Legal Capacity > Mental Disability (ID 50)
Status > Legal Capacity > Minors (ID 49)
Status > Legal Capacity > Paternal Power (ID 46)
Status > Legal Capacity > Paternal Power > Emanciaption (ID 52)
Status > Slavery (ID 8)
Status > Slavery > Manumission (ID 43)
Status > Slavery > Manumission > Paramone (ID 45)
Status > Slavery > Manumission > Pseudo Manumissions (ID 44)
Status > Slavery > Obligations of Slaves (ID 42)
Status > Slavery > Patronage/Clientela (ID 106)


## Output Format
Return ONLY a valid JSON array of objects. Do not include markdown formatting like ```json, preamble, or conversational text. Use the exact data types and structure shown in this example:

[
  {{
    "field": level_0 > level_1 > level_2,
    "field_id": 123
  }},
  {{
    "field": level_0,
    "field_id": 38
  }}
]

## Input Data
Broader Context (e.g., Full Chapter):
{broader_context}

Target Source Text, {source_name} ({language}): 
{text}

English Translation: 
{translation}
"""
# ###################################
INDEX = """## Objective
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

INDEX_W_EN = """## Objective
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

# ###################################

INDEX_V1 = """## Objective
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

INDEX_V1_CONTEXT = """## Objective
We are building an index for our corpus of legal sources.

## Instructions
1. Analyze the provided source.
2. Read the "Broader Context" provided. Use this context strictly to clarify ambiguities, identify underlying themes, or understand references within the Target Text.
3. Select terms or short phrases that are distinctive to the specific legal issues discussed in the text. 
4. Exclude very common, high-frequency words and general vocabulary (for instance: "son", "daughter", "man", "day", "house") 
5. Return the dictionary form (lemma) of the selected terms or phrases.


## Output Format
Return ONLY a valid JSON array of strings. Do not include markdown formatting, headers, or conversational text.

Example format:
["term1", "term2", "term3"]

## Input Data
Broader Context (e.g., Full Chapter):
{broader_context}

Target Source Text, {source_name} ({language}): 
{text}
"""

INDEX_W_EN_V1 = """## Objective
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

INDEX_W_EN_V1_CONTEXT = """## Objective
We are building an index for our corpus of legal sources.

## Instructions
1. Analyze the provided source and its English translation.
2. Read the "Broader Context" provided. Use this context strictly to clarify ambiguities, identify underlying themes, or understand references within the Target Text.
3. Select terms or short phrases that are distinctive to the specific legal issues discussed in the text. 
4. Exclude very common, high-frequency words and general vocabulary (for instance: "son", "daughter", "man", "day", "house") 
5. Return the dictionary form (lemma) of the selected terms or phrases.


## Output Format
Return ONLY a valid JSON array of strings. Do not include markdown formatting, headers, or conversational text.

Example format:
["term1", "term2", "term3"]

## Input Data
Broader Context (e.g., Full Chapter):
{broader_context}

Target Source Text, {source_name} ({language}): 
{text}

English Translation: 
{translation}
"""

Index_0_1 = """## Context
For a project comparing different legal cultures in the Roman world, you are tagging specific sources from each culture, in order to identify similarities and differences between them. The project includes three types of tags: judicial topics, keywords and index. You will be identifying the judicial topics of the sources. For any given source choose the most relevant judicial topics from the provided list of topics below.  

## Instructions
1. Analyze the provided {language} source.
 Interpret the legal meaning of the source, as closely as possible to the specific matters addressed in it.

2. Select from the provided source legal terminology, or specific words or short phrases that are essential to the specific legal matters, legal principles or legal situations discussed in the text. 

3. Exclude very common, high-frequency words and general vocabulary that are not pertinent to the legal matter at hand (such as particles, adjectives, adverbs, names, objects that are not relevant to the legal matter) 

4. For single words, return the dictionary form (lemma) of the selected term. For fixed phrases return its accepted form (e.g. “in potestate”) 

5. Guardrails:
Do not return words that don’t appear in the text of the provided source.
Return only words in the original language of the source.

## Examples
[1] Hebrew: Mishnah Bava Batra 8:5
האומר איש פלוני בני בכור לא יטל פי שנים, איש פלוני בני לא יירש עם אחיו, לא אמר כלום, שהתנה על מה שכתוב בתורה. המחלק נכסיו לבניו על פיו, רבה לאחד ומעט לאחד והשוה להן את הבכור, דבריו קימין. ואם אמר משום ירשה, לא אמר כלום. כתב בין בתחלה בין באמצע בין בסוף משום מתנה, דבריו קימין. האומר איש פלוני יירשני במקום שיש בת, בתי תירשני במקום שיש בן, לא אמר כלום, שהתנה על מה שכתוב בתורה. רבי יוחנן בן ברוקה אומר, אם אמר על מי שהוא ראוי לירשו, דבריו קימין. ועל מי שאין ראוי לירשו, אין דבריו קימין. הכותב את נכסיו לאחרים והניח את בניו, מה שעשה עשוי, אבל אין רוח חכמים נוחה הימנו. רבן שמעון בן גמליאל אומר, אם לא היו בניו נוהגין כשורה, זכור לטוב:

Index:
בכור, הכותב נכסיו, מחלק נכסיו, מתנה על מה שכתוב בתורה, ירושה, מתנה, נחלה, נכס, פי שנים, ראוי ליורשו, רוח חכמים נוחה הימנו
[2] Latin:  Ps. Ulpian 20.10
Filius familiae testamentum facere non potest, quoniam nihil suum habet, ut testari de eo possit. Sed divus Augustus Marcus constituit, ut filius familiae miles de eo peculio, quod in castris adquisivit, testamentum facere possit.

Index: 
castra, filius familiae, miles, peculium, testamentum, testari, testor

[3] Greek: Gnomon of Idios Logos 9
θ τ[ο]ὺς ἀπελευ[θ]έ̣ρους τῶν ἀστῶν \ἀτέκνους/ καὶ ἀδιαθέτους τελευτῶν τας κληρονο[μ]ο̣ῦσιν οἱ πάτρωνες ἢ οἱ τούτων υἱοί, ἐὰν ὦσικαὶ ἐπιδικά[ζ]ο̣νται, θυγατέρες δὲ ἢ ἄλλος τι\ς/ οὐ κληρονομήσουσι ἀλλὰ̣ ὁ φίσκος.

Index:
κληρονομέω, πάτρων, φίσκος, ἀδιάθετος, ἀπελεύθερος, ἀστός, ἄτεκνος, ἐπιδικάζομαι, ἐπιδικάζω 

## Output Format
Return ONLY a valid JSON array of strings. Do not include markdown formatting, headers, or conversational text.

Example format:
["term1", "term2", "term3"]

## Input Data
Source Text, {source_name} ({language}): 
{text}
"""

KEYWORDS_0_1_JTWC = """## Context
For a project comparing different legal cultures in the Roman world, you are tagging specific sources from each culture, in order to identify similarities and differences between them. The project includes three types of tags: judicial topics, keywords and index. You will be identifying the keywords of the sources. For any given source choose the most relevant keywords from the provided List of Keywords below. 

## Detailed Instructions
1. Analyze the provided {language} legal source{translation_note}.
	a. Interpret the legal meaning of the source, as closely as possible to the specific matters addressed in it. Identify the situation, legal issues, legal terms, formulas, legal procedures, parties and significant objects in it.
	b. Take into consideration the source’s provided textual context, only if the meaning of the source is vague.  
2. Review the provided List of Keywords.
3. Identify and return the keywords from the List of Keywords that best match the situation, legal issues, legal terms, formulas, legal procedures, parties and significant objects in the source. 
*return keywords that may be clearly inferred from the source (by an expert in the field)
*refrain from returning redundant keywords that have a very similar meaning. Among similar keywords, return the one that is most relevant to the given source. .
4. If the List of Keywords does not include all relevant keywords, suggest new keywords. 
* All new keywords MUST be in English, unless it is a widely used latin term.
*refrain from suggesting redundant keywords that have a very similar meaning.
* Assign any suggested keyword to the existing category that matches it best. If no existing category fits, use "other" for the category field.
* For suggested keywords, set the keyword_id to -1. Set the category_id to the ID of the matched category, or -1 if the category is "other".

## Examples
[1] Mishnah Bava Batra 8:7 
הכותב נכסיו לבניו, צריך שיכתב מהיום ולאחר מיתה, דברי רבי יהודה. רבי יוסי אומר, אינו צריך. הכותב נכסיו לבנו לאחר מותו, האב אינו יכול למכר, מפני שהן כתובין לבן, והבן אינו יכול למכר, מפני שהן ברשות האב. מכר האב, מכורין עד שימות. מכר הבן, אין ללוקח בהן כלום עד שימות האב. האב תולש ומאכיל לכל מי שירצה. ומה שהניח תלוש, הרי הוא של יורשין. 
Keywords: 
bequeathal clause, division of estate before death, father, son, from today and after death, gift after death, heir, inter vivos transfer, power of sale, property, usufructus, attached fruits, gathered fruits, temporary sale.  
[2] Ps. Ulpian 20.5
Ex duobus fratribus, qui in eiusdem patris potestate sunt, alter  familiae emptor, alter testis esse non potest, quoniam quod unus ex his mancipio accipit, adquirit patri, cui filius suus testis esse non debet.
Keywords: 
brother, buyer, disqualified witnesses, father, mancipatio, mancipatory will, paterfamilias, patria potestas, scale holder, son, testamentum per aes et libram, witnesses

## List of Keywords 
{keyword_list}

## Output Format
Return ONLY a valid JSON array of objects. Do not include markdown formatting like ```json, preamble, or conversational text. Use the following schema:

[
  {{
    "category": "Name of the category",
    "keyword": "Matched or suggested word",
    "suggested": true/false,
    "category_id": 123,
    "keyword_id": 456
  }}
]

## Input Data
Broader Context:
{broader_context}

Source Text, {source_name} ({language}):
{text}
{translation_section}
"""


KEYWORDS_0_1_PI = """## Context
For a project comparing different legal cultures in the Roman world, you are tagging specific sources from each culture, in order to identify similarities and differences between them. The project includes three types of tags: judicial topics, keywords and index. You will be identifying the keywords of the sources. For any given source choose the most relevant keywords from the provided list of topics below. 

## Instructions
1. Analyze the provided {language} legal source. [translation Y/N]
Divide source into its main sections. Consider standard documentary structures. 
Interpret the legal meaning of each section, as closely as possible to the specific matters addressed. Identify the situation, legal issues, legal terms, formulas, legal procedures, parties (not personal names) and significant objects in it.
Consider the possible judicial reality within which the source functioned.
2. Review the provided List of Keywords.
3. Identify and return the keywords from the List of Keywords that best match the situation, legal issues, legal terms, formulas, legal procedures, parties and significant objects in the source. 
*return keywords that may be clearly inferred from the source (by an expert in the field)
*refrain from returning redundant keywords that have a very similar meaning. Among similar keywords, return the one that is most relevant to the given source. .
4. If the List of Keywords does not include all relevant keywords, suggest new keywords. 
* All new keywords MUST be in English, unless it is a widely used latin term..
*refrain from suggesting redundant keywords that have a very similar meaning.
* Assign any suggested keyword to the existing category that matches it best. If no existing category fits, use "other" for the category field.
* For suggested keywords, set the keyword_id to -1. Set the category_id to the ID of the matched category, or -1 if the category is "other".

## Examples
[1] Egyptian papyri TM 20764
[ἔτους ̣ ̣ Αὐτοκράτορος Καίσαρος Τραιανοῦ Ἁδρια]νοῦ Σεβαστοῦ, Τῦβι ιγ, ἐν Ὀξυρύγχων πόλει τῆς Θηβαίδος, ἀγαθῇ τύχηι.
[τάδε διέθετο νοῶν καὶ φρονῶν Πεκῦσις Ἑρμοῦ τοῦ Π]εκύσιος μητρὸς Διδύμης τῆς Φιλώτου τῶν ἀπʼ Ὀξυρύγχων πόλεως ἐν ἀγυιᾷ· ἐφʼ ὃν μὲν περίειμι χρόνον ἔχειν με τὴν κατὰ τῶν ἐμῶν ἐξουσίαν
[- ca.37 - κ]α̣ὶ μεταδιατίθεσθαι. ἐὰν δὲ ἐπὶ ταύτῃ τελευτήσω τῇ διαθήκῃ, κληρονόμον ἀπολείπω τὴν θυγατέρα(*) μου Ἀμμωνοῦν μητρὸς Πτολεμᾶς, ἐὰν ζῇ, ε[ἰ δὲ]
[μή, τὴν ταύτης γενεάν, τῶν ὑπαρχόντων μοι] ἐπʼ ἀμφόδου Κρητικοῦ μερῶν κοινωνικῆς(*) οἰκίας καὶ αὐλῆς καὶ καμαρῶν. τὰ δὲ ὑπʼ ἐμοῦ ἀπολειφθησόμενα σκεύη καὶ ἔπιπλα καὶ ἐνδομενείαν καὶ εἴ τι ἄλλ[ο]
5[ἐὰν ἔχω, πάντα καταλείπω τῇ τῶν μὲν ἐμῶν τέκνω]ν μητρὶ ἐμοῦ δὲ γυναικὶ Πτολέμᾳ, ἀπελευθέρᾳ Δημητρίου Ἑρμίππου, ἐπὶ τῷ αὐτὴν ἔχειν ἐπὶ τὸν τῆς ζωῆς αὐτῆς χρόνον τὴν χρῆσιν καὶ ἐνοίκησιν καὶ ἐνοι-
[κίων ἀποφορὰν τῶν μερῶν](*)[ οἰκίας καὶ αὐλῆς καὶ καμ]αρῶν. ἐὰν δὲ συμβῇ τὴν Ἀμμωνοῦν ἄτεκνον καὶ ἀδιάθετον τελευτῆσαι, ἔσται τὰ μέρη τῶν ἐνγαίων τοῦ ὁμομητρίου αὐτῆς ἀδελφοῦ Ἀντᾶτος, ἐὰν ζῇ, εἰ δὲ μή,
[- ca.20 - καὶ μὴ ἐξεῖναι μηδενὶ ἄλλῳ π]α̣ρ̣ενχιρεῖν(*) τοῖς ὑπʼ ἐμοῦ διατεταγμένοις, ἢ τὸν παραβάντα τι τούτων ἀποτίνειν τῇ θυγατρὶ μου καὶ κληρονόμῳ Ἀμμωνοῦτι ἐπιτίμου δραχμὰς χειλίας(*) καὶ
[- ca.37 -] (hand 2) Πεκῦσις Ἑρμοῦ τοῦ Πεκύσιος καταλείπω μετὰ τελευτήν μου κληρονόμον τὴν θυγατέρα
[μου Ἀμμωνοῦν τῶν ἐπʼ ἀμφόδου Κρητι]κοῦ μερῶν οἰκίας καὶ αὐλῆς καὶ καμαρῶν· τῇ δὲ γυναικί μου Πτολέμᾳ καταλείπω πάν-
10[τα τὰ σκεύη μου καὶ ἔπιπλα καὶ ἐ]νδομενείαν καὶ εἴ τι ἄλλο αἰὰν(*) χω(*), καὶ ἐφʼ ὅσον ζῇ τὴν ἐνοίκησιν τῶν μερῶν τῆς οἰκ-
[ίας καὶ αὐλῆς καὶ καμαρῶν. ἐὰν δ]ὲ ἡ Ἀμμωνοῦς ἄτεκνος καὶ ἀδιάθετος τελευτήσῃ, ἔστω τὰ μέρη τῶν ἐνγαίων τοῦ
[ὁμομητρίου αὐτῆς ἀδελφοῦ Ἀ]ν[τ]ᾶτος ὡς πρόκιται. εἰμὶ ἐτῶν τεσσαράκοντα τεσσάρων , οὐλὴ τραχήλῳ ἐξ ἀριστερῶν,
[καὶ ἔστι μου ἡ σφραγὶς Ἄμ]μωνος.(*). (hand 3) Σαραπίων Σαραπίωνος τοῦ Διονυσίου ἀπὸ τῆς αὐτῆς πόλεως μαρτυρῶ τῇ τοῦ Πεκυσις(*) διαθήκῃ, καὶ
[εἰμὶ ἐτῶν ̣ ̣, οὐλὴ ̣ ̣ ̣ ̣ ̣ ̣, καὶ ἔστι μου ἡ σφ]ραγὶς Διονύσου. (hand 4) Ἑκάτων Σαραπίωνος τοῦ Ἑκάτωνος ἀπὸ τῆς αὐτῆς πόλεως μαρτυρῶ τῇ τοῦ Πεκύσιος διαθήκῃ, καὶ εἰμὶ
15[ἐτῶν ̣ ̣, οὐλὴ - ca.13 - , καὶ ἔστι μο]υ ἡ σφραγὶς Σαράπιδος. (hand 5) Παποντὼς Διογένους τοῦ Παποντῶτος ἀπὸ τῆς αὐτῆς πόλεως μαρτυρῶ τῇ τοῦ Πεκύσιος
[διαθήκῃ, καὶ εἰμὶ ἐτῶν ̣ ̣ ̣ ̣ ̣ ̣ ̣, καὶ] ἔστιν μου ἡ σφραγὶς Διὸς ἐπʼ ἀέτῳ(*). (hand 6) Ζωίλος Ζωίλου τοῦ Πανεχώτου τῶν ἀπὸ τῆς αὐτ-
[ῆς πόλεως μαρτυρῶ τῇ τοῦ Π]εκύσεος διαθήκῃ, καὶ ἰμὶ(*) ἐτῶν τεσσαράκοντα ὀκτὼ , \οὐλὴ/ πήχι(*) ἀριστερῷ, ἡ
[δὲ σφραγίς μού ἐστιν ̣ ̣ ̣ ̣ ̣ ̣ Ἁ]ρποκράτου ἐπὶ κιβωρτωι. (hand 7) Ἡρᾶς ὁ καὶ Σάιος Ζηνᾶτος τοῦ Ἡρᾶτος ἀπὸ τῆς αὐτῆς πόλεως μαρτυρῶι(*) τῇ τοῦ Πεκύσιος
[διαθήκῃ, καὶ εἰμὶ ἐτῶν - ca.10 -, οὐλὴ ἀντικνημ]ίωι δεξιῶι, καὶ ἔστι μου ἡ σφραγὶ[ς π]ρ[ο]τομὴ(*) φιλ[ο]σόφου. (hand 8) Διονύσιος Διον[υσ]ίου τ[ο]ῦ Διογένους ἀπὸ τῆς αὐτῆ[ς] πόλεως μαρτ[υ]ρῶ
20[τῇ τοῦ Πεκύσιος διαθήκῃ, καὶ εἰμὶ] ἐτῶν τεσσαράκοντα ἕξ , οὐλὴ παρὰ κρόταφον δεξιόν, καὶ ἔστι μου ἡ σφραγὶς Διονυσοπλάτωνος. (hand 9) μν̣η̣μ̣(ονείου)(*) Ὀξυρ(ύγχων) πόλ(εως).
[ἔτους ̣ ̣ Αὐτοκράτορος Καί]σαρος Τραιανοῦ Ἁδριανοῦ Σεβαστοῦ(*), Τῦβι ιγ.
[⁦ -ca.?- ⁩ διαθήκη Πεκύσιος Ἑρ]μοῦ τοῦ Πεκύσιος μητρὸ(ς) Διδύμης Φιλώτου ἀπʼ Ὀξ(υρύγχων) π[ό]λ(εως).
Translation: The ... year of the Emperor Trajanus Hadrianus Augustus, Tybi 13, at Oxyrhynchus in the Thebaid; for good luck. This is the will, made in the street, of Pekusis, son of Hermes and Didyme, daughter of Philotas, an inhabitant of Oxyrhynchus, being sane and in his right mind. So long as I survive, I am to have power over my property, to ... and to alter my will. But if I die with this will unchanged, I leave my daughter Ammonous whose mother is Ptolema, if she survive me, but if not, then her children, heir to my shares in the common house, court and rooms situated in the Cretan quarter. All the furniture, movables and household stock and other property whatsoever that I shall leave, I bequeath to the mother of my children and my wife, Ptolema, the freedwoman of Demetrius, son of Hermippus, with the condition that she shall have for her lifetime the right of using, dwelling in, and building in the said house, court and rooms. If Ammonous should die without children and intestate, the share of the fixtures shall belong to her half-brother on the mother’s side, Antas, if he survive, but if not, to.... No one shall violate the terms of this my will under pain of paying to my daughter and heir Ammonous a fine of 1000 drachmae and (to the treasury an equal sum?) 
Keywords: 
Right of habitation for the widow, testamentary heir, bequeathal clause, currency, daughter, fine, freedperson, heir, house in the city, immovable property, movable property, penalty clause, penalty for breach of will, property, specification of assets, substitutio vulgaris, ordinary substitution, testament formula, testator, usufructus, witnesses, woman
[2] Greek Inscription SEG 9.7 (Will of Ptolemy VIII Euergetes II)
Ἔτους πεντεκαιδεκάτου, μηνὸς Λώιου.Ἀγαθῆι τύχηι. Τάδε διέθετο βασιλεὺς
Πτολεμαῖος βασιλέως Πτολεμαίου
καὶ βασιλίσσης Κλεοπάτρας, θεῶν
Ἐπιφανῶν, ὁ νεώτερος· ὧν καὶ τὰ ἀντίγραφα
εἰς Ῥώμην ἐξαπέσταλται. — Εἴη μέν μοι
μετὰ τῆς τῶν θεῶν εὐμενείας μετελθεῖν
καταξίως τοὺς συστησαμένους ἐπί με
τὴν ἀνόσιον ἐπιβουλὴν καὶ προελομένους
μὴ μόνον ⟦oν⟧ τῆς βασιλείας, ἀλλὰ καὶ
τοῦ ζῆν στερῆσαί με· ἐὰν δέ τι συμβαίνηι
τῶν κατ’ ἄνθρωπον πρότερον ἢ διαδόχους
ἀπολιπεῖν τῆς βασιλείας, καταλείπω
Ῥωμαίοις τὴν καθήκουσάν μοι βασιλείαν,
οἷς ἀπ’ ἀρχῆς τήν τε φιλίαν καὶ τὴν
συμμαχίαν γνησίως συντετήρηκα·
τοῖς δ’ αὐτοῖς παρακατατίθεμαι τὰ πράγματα
συντηρεῖν, ἐνευχόμενος κατά τε τῶν θεῶν
πάντων καὶ τῆς ἑαυτῶν εὐδοξίας, ἐάν τινες
ἐπίωσιν ἢ ταῖς πόλεσιν ἢ τῆι χώραι, βοηθεῖν
κατὰ τὴν ψιλίαν καὶ συμμαχίαν τὴν ⟦πρὸς⟧
πρὸς ἀλλήλους ἡμῖν γενομένην καὶ τὸ
δίκαιον παντὶ σθένει.
— Μάρτυρας δὲ τούτων ποιοῦμαι Δία τε τὸν
Καπετώλιον καὶ τοὺς Μεγάλους θεοὺς
καὶ τὸν Ἥλιον καὶ τὸν Ἀρχηγέτην Ἀπόλλωνα,
παρ’ ὧι καὶ τὰ περὶ τούτων ἀνιέρωται γράμματα.
Τύχηι τῆι ἀγαθῆι
Keywords: 
king, testamentary heir, deposit, heir, oath clauses, property, public document, testament formula, conditional bequeathal, testamentary intent, witnesses, bequeathal of kingdom, political alliance, divine witnesses. 

## List of Keywords 
{keyword_list}

## Output Format
Return ONLY a valid JSON array of objects. Do not include markdown formatting like ```json, preamble, or conversational text. Use the following schema:

[
  {{
    "category": "Name of the category",
    "keyword": "Matched or suggested word",
    "suggested": true/false,
    "category_id": 123,
    "keyword_id": 456
  }}
]

## Input Data
Source Text, {reference_name} ({language}):
{text}

English Translation:
{translation}

"""


TOPICS_0_1_JTWC = """## Context
For a project comparing different legal cultures in the Roman world, you are tagging specific sources from each culture, in order to identify similarities and differences between them. The project includes three types of tags: judicial topics, keywords and index. You will be identifying the judicial topics of the sources. For any given source choose the most relevant judicial topics from the provided list of topics below.  

## Instructions
1. Analyze the provided {language} legal source. [translation Y/N]
	a. Interpret the legal meaning of the source, as closely as possible to the specific matters it addresses.
	b. Take into consideration its provided textual context, as long as it is directly relevant.
2. Review the provided Judicial Topic Hierarchy.
3. Identify and return the fields from the Judicial Topic Hierarchy that best match the specific content of the provided text.
Select the lowest level possible that is relevant; otherwise, choose a relevant field from the level above.
If none of the fields in the hierarchy adequately cover the source, return an empty list.  
## Examples
[1] Mishnah Bava Batra 8:1
יש נוחלין ומנחילין, ויש נוחלין ולא מנחילין, מנחילין ולא נוחלין, לא נוחלין ולא מנחילין. ואלו נוחלין ומנחילין, האב את הבנים והבנים את האב והאחין מן האב, נוחלין ומנחילין. האיש את אמו והאיש את אשתו, ובני אחיות, נוחלין ולא מנחילין. האשה את בניה והאשה את בעלה ואחי האם, מנחילין ולא נוחלין. והאחים מן האם, לא נוחלין ולא מנחילין:
Topics: 
Classes of Heirs, Women Inheritance Rights

[2] Tosefta Bava Batra 7:17
כתב ירושה מלמטה ומתנה מלמעלה, ירושה מלמעלה ומתנה מלמטה, ירושה מכן ומכן ומתנה באמצע, הואיל והזכיר שום מתנה, דבריו קיימין.
Topics:
Deeds of Gift as Wills, Forms of Will
[3] Ps. Ulpian 20.7
Mutus, surdus, furiosus, pupillus, femina neque familiae emptor esse, neque testis libripensve fieri potest. 
Topics:
Legal Capacity, Legal Capacity of Women, Mental Disability, Minors, Testaments

## Judicial Topic Hierarchy 
Courts and Procedure (ID 4)
Courts and Procedure > Appeal (ID 20)
Courts and Procedure > Courts (ID 16)
Courts and Procedure > Courts > Arbitration (ID 100)
Courts and Procedure > Courts > Choice of Judges (ID 99)
Courts and Procedure > Evidence (ID 17)
Courts and Procedure > Evidence > Archiving (ID 109)
Courts and Procedure > Evidence > Documentary Evidence (ID 102)
Courts and Procedure > Evidence > Witnesses (ID 101)
Courts and Procedure > Execution (ID 19)
Courts and Procedure > Oaths (ID 18)
Courts and Procedure > Procedure (ID 121)
Courts and Procedure > Procedure > Confession (ID 122)
Obligations (ID 3)
Obligations > Consensual Contract (ID 14)
Obligations > Consensual Contract > Deposit (ID 79)
Obligations > Consensual Contract > Deposit > Responsibility of depositee (ID 90)
Obligations > Consensual Contract > Letting and Hiring (ID 78)
Obligations > Consensual Contract > Letting and Hiring > Emphyteusis (ID 88)
Obligations > Consensual Contract > Letting and Hiring > Labour (ID 85)
Obligations > Consensual Contract > Letting and Hiring > Land Tenancy (ID 86)
Obligations > Consensual Contract > Letting and Hiring > Misthosis (ID 89)
Obligations > Consensual Contract > Loan and Pledge (ID 74)
Obligations > Consensual Contract > Loan and Pledge > Interest (ID 80)
Obligations > Consensual Contract > Loan and Pledge > Real security (ID 81)
Obligations > Consensual Contract > Loan and Pledge > Transfer of Debt (ID 82)
Obligations > Consensual Contract > Mandate (ID 77)
Obligations > Consensual Contract > Partnership (ID 76)
Obligations > Consensual Contract > Sale (ID 75)
Obligations > Consensual Contract > Sale > Fraud (ID 84)
Obligations > Consensual Contract > Sale > Warrenty (ID 83)
Obligations > Delicts (ID 15)
Obligations > Delicts > Damage (ID 93)
Obligations > Delicts > Damage > Indirect cause (ID 98)
Obligations > Delicts > Damage > wild Animals (ID 97)
Obligations > Delicts > Theft (ID 91)
Obligations > Delicts > Theft > In deposit (ID 96)
Obligations > Delicts > Theft > Intention (ID 95)
Obligations > Delicts > Theft > Return of Object (ID 94)
Obligations > Delicts > Wild Animals (ID 92)
Obligations > Verbal Contracts (ID 107)
Obligations > Verbal Contracts > Stipulation (ID 108)
Property (ID 2)
Property > Acquisition (ID 10)
Property > Acquisition > Conveyence (ID 55)
Property > Acquisition > Conveyence > Classification of Things (ID 60)
Property > Acquisition > Conveyence > Delivery (ID 62)
Property > Acquisition > Conveyence > Formal act (ID 61)
Property > Acquisition > Conveyence > Payment (ID 63)
Property > Acquisition > Conveyence > Usucapio (ID 64)
Property > Acquisition > Occupation (ID 56)
Property > Acquisition > Occupation > Abandoned Property (ID 68)
Property > Acquisition > Occupation > Accession (ID 69)
Property > Acquisition > Occupation > From Natural state (ID 65)
Property > Acquisition > Occupation > Specification (ID 67)
Property > Acquisition > Occupation > Superficies (ID 66)
Property > Acquisition > Usufruct (ID 57)
Property > Neighbors (ID 13)
Property > Possession (ID 11)
Property > Servitudes (ID 12)
Property > Servitudes > Personal (ID 59)
Property > Servitudes > Personal > Usufruct (ID 73)
Property > Servitudes > Praedial (ID 58)
Property > Servitudes > Praedial > Rights of Light (ID 72)
Property > Servitudes > Praedial > Rights of Water (ID 71)
Property > Servitudes > Praedial > Rights of Way (ID 70)
Public Law (ID 110)
Public Law > Administrative Law (ID 112)
Public Law > Administrative Law > Archive (ID 114)
Public Law > Administrative Law > Judiciary (ID 113)
Public Law > Administrative Law > Municipal Government  (ID 115)
Public Law > Administrative Law > Provincial Government  (ID 123)
Public Law > Taxation (ID 111)
Status (ID 1)
Status > Citizenship (ID 7)
Status > Citizenship > Captivity (ID 41)
Status > Citizenship > Freed Persons (ID 105)
Status > Citizenship > Naturalization (ID 40)
Status > Family (ID 5)
Status > Family > Divorce (ID 22)
Status > Family > Marital Arrangments (ID 23)
Status > Family > Marital Arrangments > Dowry (ID 24)
Status > Family > Marital Arrangments > Property of Women (ID 27)
Status > Family > Marital Arrangments > Provision for Wife (ID 25)
Status > Family > Marital Arrangments > Provisions for Children (ID 26)
Status > Family > Marriage (ID 21)
Status > Family > Marriage > Acts of Marriage (ID 117)
Status > Family > Marriage > Legitimate Children (ID 118)
Status > Inheritance (ID 6)
Status > Inheritance > Debts and Obligations (ID 119)
Status > Inheritance > Intestacy (ID 28)
Status > Inheritance > Intestacy > Classes of Heirs (ID 30)
Status > Inheritance > Intestacy > Father Inheritance Rights  (ID 103)
Status > Inheritance > Intestacy > Firstborn Inheritance Rights (ID 104)
Status > Inheritance > Intestacy > Women Inheritance rights (ID 31)
Status > Inheritance > Testaments (ID 29)
Status > Inheritance > Testaments > Accaptance of will (ID 35)
Status > Inheritance > Testaments > Capacity (ID 34)
Status > Inheritance > Testaments > Causa Mortis (ID 32)
Status > Inheritance > Testaments > Deeds of Gift as Wills (ID 37)
Status > Inheritance > Testaments > Disinheritance (ID 33)
Status > Inheritance > Testaments > Forms of Will (ID 36)
Status > Inheritance > Testaments > Legacies (ID 38)
Status > Inheritance > Testaments > Trust (ID 39)
Status > Legal Capacity (ID 9)
Status > Legal Capacity > Adoption (ID 47)
Status > Legal Capacity > Guardianship (ID 48)
Status > Legal Capacity > Guardianship > Guardianship over Minors (ID 53)
Status > Legal Capacity > Guardianship > Guardianship over Women (ID 54)
Status > Legal Capacity > Infamy (ID 51)
Status > Legal Capacity > Legal Capcity of Women (ID 120)
Status > Legal Capacity > Mental Disability (ID 50)
Status > Legal Capacity > Minors (ID 49)
Status > Legal Capacity > Paternal Power (ID 46)
Status > Legal Capacity > Paternal Power > Emanciaption (ID 52)
Status > Slavery (ID 8)
Status > Slavery > Manumission (ID 43)
Status > Slavery > Manumission > Paramone (ID 45)
Status > Slavery > Manumission > Pseudo Manumissons (ID 44)
Status > Slavery > Obligations of Slaves (ID 42)
Status > Slavery > Patronage/Clientela (ID 106)

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
Broader Context:
{broader_context}

Source Text, {reference_name} ({language}):
{text}

English Translation:
{translation}

"""


TOPICS_0_1_PI = """## Context
For a project comparing different legal cultures in the Roman world, you are tagging specific sources from each culture, in order to identify similarities and differences between them. The project includes three types of tags: judicial topics, keywords and index. You will be identifying the judicial topics of the sources. For any given source choose the most relevant judicial topics from the provided list of topics below.  

## Instructions
1. Analyze the provided {language} source. [translation Y/N]
Divide text into its main sections. Consider standard documentary structures. 
Interpret the legal meaning of each section, as closely as possible to the specific matters addressed.
Consider the possible judicial reality within which the text functioned.
2. Review the provided Judicial Topic Hierarchy.
3. Identify and return the fields from the Judicial Topic Hierarchy that best match the specific contents of the provided text.
Select the lowest level possible that is relevant; otherwise, choose a relevant field from the level above.
If none of the fields in the hierarchy adequately cover the source, return an empty list.  
## Examples
[1] Egyptian papyri TM 20764
[ἔτους ̣ ̣ Αὐτοκράτορος Καίσαρος Τραιανοῦ Ἁδρια]νοῦ Σεβαστοῦ, Τῦβι ιγ, ἐν Ὀξυρύγχων πόλει τῆς Θηβαίδος, ἀγαθῇ τύχηι.
[τάδε διέθετο νοῶν καὶ φρονῶν Πεκῦσις Ἑρμοῦ τοῦ Π]εκύσιος μητρὸς Διδύμης τῆς Φιλώτου τῶν ἀπʼ Ὀξυρύγχων πόλεως ἐν ἀγυιᾷ· ἐφʼ ὃν μὲν περίειμι χρόνον ἔχειν με τὴν κατὰ τῶν ἐμῶν ἐξουσίαν
[- ca.37 - κ]α̣ὶ μεταδιατίθεσθαι. ἐὰν δὲ ἐπὶ ταύτῃ τελευτήσω τῇ διαθήκῃ, κληρονόμον ἀπολείπω τὴν θυγατέρα(*) μου Ἀμμωνοῦν μητρὸς Πτολεμᾶς, ἐὰν ζῇ, ε[ἰ δὲ]
[μή, τὴν ταύτης γενεάν, τῶν ὑπαρχόντων μοι] ἐπʼ ἀμφόδου Κρητικοῦ μερῶν κοινωνικῆς(*) οἰκίας καὶ αὐλῆς καὶ καμαρῶν. τὰ δὲ ὑπʼ ἐμοῦ ἀπολειφθησόμενα σκεύη καὶ ἔπιπλα καὶ ἐνδομενείαν καὶ εἴ τι ἄλλ[ο]
5[ἐὰν ἔχω, πάντα καταλείπω τῇ τῶν μὲν ἐμῶν τέκνω]ν μητρὶ ἐμοῦ δὲ γυναικὶ Πτολέμᾳ, ἀπελευθέρᾳ Δημητρίου Ἑρμίππου, ἐπὶ τῷ αὐτὴν ἔχειν ἐπὶ τὸν τῆς ζωῆς αὐτῆς χρόνον τὴν χρῆσιν καὶ ἐνοίκησιν καὶ ἐνοι-
[κίων ἀποφορὰν τῶν μερῶν](*)[ οἰκίας καὶ αὐλῆς καὶ καμ]αρῶν. ἐὰν δὲ συμβῇ τὴν Ἀμμωνοῦν ἄτεκνον καὶ ἀδιάθετον τελευτῆσαι, ἔσται τὰ μέρη τῶν ἐνγαίων τοῦ ὁμομητρίου αὐτῆς ἀδελφοῦ Ἀντᾶτος, ἐὰν ζῇ, εἰ δὲ μή,
[- ca.20 - καὶ μὴ ἐξεῖναι μηδενὶ ἄλλῳ π]α̣ρ̣ενχιρεῖν(*) τοῖς ὑπʼ ἐμοῦ διατεταγμένοις, ἢ τὸν παραβάντα τι τούτων ἀποτίνειν τῇ θυγατρὶ μου καὶ κληρονόμῳ Ἀμμωνοῦτι ἐπιτίμου δραχμὰς χειλίας(*) καὶ
[- ca.37 -] (hand 2) Πεκῦσις Ἑρμοῦ τοῦ Πεκύσιος καταλείπω μετὰ τελευτήν μου κληρονόμον τὴν θυγατέρα
[μου Ἀμμωνοῦν τῶν ἐπʼ ἀμφόδου Κρητι]κοῦ μερῶν οἰκίας καὶ αὐλῆς καὶ καμαρῶν· τῇ δὲ γυναικί μου Πτολέμᾳ καταλείπω πάν-
10[τα τὰ σκεύη μου καὶ ἔπιπλα καὶ ἐ]νδομενείαν καὶ εἴ τι ἄλλο αἰὰν(*) χω(*), καὶ ἐφʼ ὅσον ζῇ τὴν ἐνοίκησιν τῶν μερῶν τῆς οἰκ-
[ίας καὶ αὐλῆς καὶ καμαρῶν. ἐὰν δ]ὲ ἡ Ἀμμωνοῦς ἄτεκνος καὶ ἀδιάθετος τελευτήσῃ, ἔστω τὰ μέρη τῶν ἐνγαίων τοῦ
[ὁμομητρίου αὐτῆς ἀδελφοῦ Ἀ]ν[τ]ᾶτος ὡς πρόκιται. εἰμὶ ἐτῶν τεσσαράκοντα τεσσάρων , οὐλὴ τραχήλῳ ἐξ ἀριστερῶν,
[καὶ ἔστι μου ἡ σφραγὶς Ἄμ]μωνος.(*). (hand 3) Σαραπίων Σαραπίωνος τοῦ Διονυσίου ἀπὸ τῆς αὐτῆς πόλεως μαρτυρῶ τῇ τοῦ Πεκυσις(*) διαθήκῃ, καὶ
[εἰμὶ ἐτῶν ̣ ̣, οὐλὴ ̣ ̣ ̣ ̣ ̣ ̣, καὶ ἔστι μου ἡ σφ]ραγὶς Διονύσου. (hand 4) Ἑκάτων Σαραπίωνος τοῦ Ἑκάτωνος ἀπὸ τῆς αὐτῆς πόλεως μαρτυρῶ τῇ τοῦ Πεκύσιος διαθήκῃ, καὶ εἰμὶ
15[ἐτῶν ̣ ̣, οὐλὴ - ca.13 - , καὶ ἔστι μο]υ ἡ σφραγὶς Σαράπιδος. (hand 5) Παποντὼς Διογένους τοῦ Παποντῶτος ἀπὸ τῆς αὐτῆς πόλεως μαρτυρῶ τῇ τοῦ Πεκύσιος
[διαθήκῃ, καὶ εἰμὶ ἐτῶν ̣ ̣ ̣ ̣ ̣ ̣ ̣, καὶ] ἔστιν μου ἡ σφραγὶς Διὸς ἐπʼ ἀέτῳ(*). (hand 6) Ζωίλος Ζωίλου τοῦ Πανεχώτου τῶν ἀπὸ τῆς αὐτ-
[ῆς πόλεως μαρτυρῶ τῇ τοῦ Π]εκύσεος διαθήκῃ, καὶ ἰμὶ(*) ἐτῶν τεσσαράκοντα ὀκτὼ , \οὐλὴ/ πήχι(*) ἀριστερῷ, ἡ
[δὲ σφραγίς μού ἐστιν ̣ ̣ ̣ ̣ ̣ ̣ Ἁ]ρποκράτου ἐπὶ κιβωρτωι. (hand 7) Ἡρᾶς ὁ καὶ Σάιος Ζηνᾶτος τοῦ Ἡρᾶτος ἀπὸ τῆς αὐτῆς πόλεως μαρτυρῶι(*) τῇ τοῦ Πεκύσιος
[διαθήκῃ, καὶ εἰμὶ ἐτῶν - ca.10 -, οὐλὴ ἀντικνημ]ίωι δεξιῶι, καὶ ἔστι μου ἡ σφραγὶ[ς π]ρ[ο]τομὴ(*) φιλ[ο]σόφου. (hand 8) Διονύσιος Διον[υσ]ίου τ[ο]ῦ Διογένους ἀπὸ τῆς αὐτῆ[ς] πόλεως μαρτ[υ]ρῶ
20[τῇ τοῦ Πεκύσιος διαθήκῃ, καὶ εἰμὶ] ἐτῶν τεσσαράκοντα ἕξ , οὐλὴ παρὰ κρόταφον δεξιόν, καὶ ἔστι μου ἡ σφραγὶς Διονυσοπλάτωνος. (hand 9) μν̣η̣μ̣(ονείου)(*) Ὀξυρ(ύγχων) πόλ(εως).
[ἔτους ̣ ̣ Αὐτοκράτορος Καί]σαρος Τραιανοῦ Ἁδριανοῦ Σεβαστοῦ(*), Τῦβι ιγ.
[⁦ -ca.?- ⁩ διαθήκη Πεκύσιος Ἑρ]μοῦ τοῦ Πεκύσιος μητρὸ(ς) Διδύμης Φιλώτου ἀπʼ Ὀξ(υρύγχων) π[ό]λ(εως).
Translation: The ... year of the Emperor Trajanus Hadrianus Augustus, Tybi 13, at Oxyrhynchus in the Thebaid; for good luck. This is the will, made in the street, of Pekusis, son of Hermes and Didyme, daughter of Philotas, an inhabitant of Oxyrhynchus, being sane and in his right mind. So long as I survive, I am to have power over my property, to ... and to alter my will. But if I die with this will unchanged, I leave my daughter Ammonous whose mother is Ptolema, if she survive me, but if not, then her children, heir to my shares in the common house, court and rooms situated in the Cretan quarter. All the furniture, movables and household stock and other property whatsoever that I shall leave, I bequeath to the mother of my children and my wife, Ptolema, the freedwoman of Demetrius, son of Hermippus, with the condition that she shall have for her lifetime the right of using, dwelling in, and building in the said house, court and rooms. If Ammonous should die without children and intestate, the share of the fixtures shall belong to her half-brother on the mother’s side, Antas, if he survive, but if not, to.... No one shall violate the terms of this my will under pain of paying to my daughter and heir Ammonous a fine of 1000 drachmae and (to the treasury an equal sum?) 
Topics: 
Testaments, Forms of Will
[2] P.  Yadin 19
[⁦ -ca.?- ⁩] ἥ[μι]σ̣υ [⁦ -ca.?- ⁩]
[⁦ -ca.?- ⁩ διαθετ]ῶ̣ν̣, ν̣[ό]τ̣[ου]
[ἀγορά,] βορ[ρᾶ ⁦ -ca.?- ⁩]α̣ι τ̣α̣ύ̣τ̣η̣ν̣ [κ]αὶ ἄλλη̣[ν]
[⁦ -ca.?- ⁩] ̣ δι̣α̣θή̣[κ]ην κυ-
5[⁦ -ca.?- ⁩ σκάπ]τειν β̣[ο]θ̣α̣νειν(*)
[⁦ -ca.?- ⁩] ̣ ̣ ̣ ̣[- ca.15 -παρ]α̣ν̣[γ]ε̣ι̣ει(*) Σελαμψ̣[ι-]
[οῦς τῷ αὐτῷ Ἰούδατι τευχίσει αὐτὴν διὰ δημοσί]ω̣ν̣. ⁦ vac. ? ⁩
r,int,Fr??
1[⁦ -ca.?- ⁩]π̣ασι οις αι[⁦ -ca.?- ⁩]
[⁦ -ca.?- ⁩]η̣[⁦ -ca.?- ⁩]
ἐπ̣[ὶ ὑπ]ά̣τ̣[ων Πουβλίου Με]τ̣ε̣[ιλίου] Ν̣έ̣π̣ω̣[τος] τὸ δ[εύτερο]ν κ̣[αὶ Μάρ]κου Ἀνν̣ίου
Λ̣ί̣β̣ω̣ν̣ο̣ς̣ π̣ρ̣[ὸ ἑκ]κ̣α̣ί̣δ̣ε̣κ̣α̣ κ̣α̣λ̣α̣ν̣δ̣ῶ̣ν̣ Μαίων, κ̣α̣τ̣ὰ̣ [τὸν] ἀ̣ρ̣[ιθ]μ̣ὸ̣ν̣ [τῆς] ν̣έ̣α̣ς̣ ἐ̣π̣α̣ρ̣-
10[χείας ἔτους τρίτου εἰ]κ̣[οστο]ῦ̣ Ξ̣α̣νδικοῦ ἕκ[τ]ῃ καὶ εἰκάδ[ι], ἐν̣ Μ̣α̣ω̣ζας(*) τῆς πε-
[ρὶ Ζοα]ρ̣α, [δι]έ̣θ̣ε̣τ̣[ο Ἰο]ύ̣δ̣ας Ἐ̣λ[αζά]ρου Χθουσ[ίω]νο̣ς̣ Ἠ̣νγ̣α̣δη̣[νὸ]ς̣(*) οἰκῶν ἐν
Μαωζᾶς(*) [Σελ]αμψιο̣ῦ̣ς θυ[γατ]ερ̣(*) π̣ά̣ν̣[τα τὰ ὑ]πάρ[χον]τ̣α αὐ̣[τ]ῷ [ἐ]ν Ἠνγ̣αδῆς(*)
ἥ[μισ]υ̣ α[ὐ]λ̣ῆς ̣ ̣ ̣ ̣[ ̣ ̣]ρ̣αν [ ̣ ̣]ν̣α̣γ̣ωγ̣[ ̣ ̣]ν̣η̣ ̣[ ̣] ̣ν̣οτ̣ο̣[ ̣ ̣ ̣]ν ἥμισυ οἰ-
κοιμάτων̣(*) καὶ ὑπερωαις(*) ἐνο[υ]σι(*) χωρὶς αὐ̣λῆς μικκῆς π̣αλ̣εαν(*) ἐ̣νγὺς(*)
15τῆς α̣ὐτ̣[ῆ]ς̣ αὐλῆς, καὶ τ[ὸ] ἄλλ̣ο̣ ἥμισυ τῆς αὐλῆς καὶ οἰκο̣ιμ̣ά̣των(*) διέθε-
το̣ ̣ ̣[ ̣ Ἰ]ο̣ύ̣δας τ̣ῇ αὐ̣[τ]ῇ [Σελ]α̣μψιοῦ[ς] μετὰ τὸ αὐτὸ[ν] τ̣ε<λε>υτῆσαι, ὧν
γείτ̣ω̣νες̣(*) [τ]ῆ̣ς αὐλῆς καὶ οἰκοιμά[των](*)[ ἀν]α̣τολῶ̣ν Ἰησοῦ Μαδδαρωνα
καὶ αὐρίχω̣ρον(*), δυσμῶν ὁ διεθετῶν(*), νότου ἀγ̣ορά, βορρᾶ ὁδός, περὶ δὲ
πλάνης γειτνιῶν οὐ μεθάξει(*) Σελαμψιοῦς, σὺν εἰσόδοις κ̣α̣ὶ ἐξόδο̣ις, πλίν-
20θοις, δόκωσι(*), θυρίαις(*), θυρίσι, καὶ τοῖς ἐνοῦσι πανταίοις(*), ὥστε ἔχειν τὴν προ-
γ̣εγ̣ρ̣α̣μ̣μ̣[έ]ν̣ην Σελαμψ̣ιοῦ[ς] τ̣ὸ ἥμισυ̣ τῆς προγεγραμμένης αὐ-
λῆς καὶ οἰ[κη]μ̣[ά]τ̣ω̣ν̣ ἀ[πὸ τῆς σήμερον καὶ τὸ] ἄ̣λ̣λο ἥμ[ι]σ̣υ̣ μετὰ τὲ(*) τε-
λ̣[ευ]τ̣ῆσα̣ι̣ τοῦ αὐτοῦ Ἰούδα κ̣υρίω[ς](*)[ καὶ βε]β̣αίω̣ς εἰς τὸν ἅπ̣α̣ντα χρόνον̣,
[οἰκ]οδομεῖν, ὑπερ<αίρ>ειν̣, ὑψε̣ῖ̣ν(*), σκάπτε̣ι̣ν, βοθανειν(*), κτᾶσθαι, χρᾶσθαι, πω-
25λεῖν, διοικεῖν, τ̣ρόπῳ ᾧ ἂ̣ν̣ α̣ἱρῆ̣<ται>, π̣ά̣ν̣τ̣α κ̣ύ̣ρ̣ι̣α̣ καὶ βέβαια. ὅταν δὲ
παρα̣νγείλει(*) Σελα<μ>ψιοῦς τῷ αὐτῷ Ἰούδατι, τευχιζζει(*) αὐτὴν διὰ δημο-
σίων.
Topics: 
Deeds of Gift as Wills, Testaments, 

## Judicial Topic Hierarchy 
Courts and Procedure (ID 4)
Courts and Procedure > Appeal (ID 20)
Courts and Procedure > Courts (ID 16)
Courts and Procedure > Courts > Arbitration (ID 100)
Courts and Procedure > Courts > Choice of Judges (ID 99)
Courts and Procedure > Evidence (ID 17)
Courts and Procedure > Evidence > Archiving (ID 109)
Courts and Procedure > Evidence > Documentary Evidence (ID 102)
Courts and Procedure > Evidence > Witnesses (ID 101)
Courts and Procedure > Execution (ID 19)
Courts and Procedure > Oaths (ID 18)
Courts and Procedure > Procedure (ID 121)
Courts and Procedure > Procedure > Confession (ID 122)
Obligations (ID 3)
Obligations > Consensual Contract (ID 14)
Obligations > Consensual Contract > Deposit (ID 79)
Obligations > Consensual Contract > Deposit > Responsibility of depositee (ID 90)
Obligations > Consensual Contract > Letting and Hiring (ID 78)
Obligations > Consensual Contract > Letting and Hiring > Emphyteusis (ID 88)
Obligations > Consensual Contract > Letting and Hiring > Labour (ID 85)
Obligations > Consensual Contract > Letting and Hiring > Land Tenancy (ID 86)
Obligations > Consensual Contract > Letting and Hiring > Misthosis (ID 89)
Obligations > Consensual Contract > Loan and Pledge (ID 74)
Obligations > Consensual Contract > Loan and Pledge > Interest (ID 80)
Obligations > Consensual Contract > Loan and Pledge > Real security (ID 81)
Obligations > Consensual Contract > Loan and Pledge > Transfer of Debt (ID 82)
Obligations > Consensual Contract > Mandate (ID 77)
Obligations > Consensual Contract > Partnership (ID 76)
Obligations > Consensual Contract > Sale (ID 75)
Obligations > Consensual Contract > Sale > Fraud (ID 84)
Obligations > Consensual Contract > Sale > Warrenty (ID 83)
Obligations > Delicts (ID 15)
Obligations > Delicts > Damage (ID 93)
Obligations > Delicts > Damage > Indirect cause (ID 98)
Obligations > Delicts > Damage > wild Animals (ID 97)
Obligations > Delicts > Theft (ID 91)
Obligations > Delicts > Theft > In deposit (ID 96)
Obligations > Delicts > Theft > Intention (ID 95)
Obligations > Delicts > Theft > Return of Object (ID 94)
Obligations > Delicts > Wild Animals (ID 92)
Obligations > Verbal Contracts (ID 107)
Obligations > Verbal Contracts > Stipulation (ID 108)
Property (ID 2)
Property > Acquisition (ID 10)
Property > Acquisition > Conveyence (ID 55)
Property > Acquisition > Conveyence > Classification of Things (ID 60)
Property > Acquisition > Conveyence > Delivery (ID 62)
Property > Acquisition > Conveyence > Formal act (ID 61)
Property > Acquisition > Conveyence > Payment (ID 63)
Property > Acquisition > Conveyence > Usucapio (ID 64)
Property > Acquisition > Occupation (ID 56)
Property > Acquisition > Occupation > Abandoned Property (ID 68)
Property > Acquisition > Occupation > Accession (ID 69)
Property > Acquisition > Occupation > From Natural state (ID 65)
Property > Acquisition > Occupation > Specification (ID 67)
Property > Acquisition > Occupation > Superficies (ID 66)
Property > Acquisition > Usufruct (ID 57)
Property > Neighbors (ID 13)
Property > Possession (ID 11)
Property > Servitudes (ID 12)
Property > Servitudes > Personal (ID 59)
Property > Servitudes > Personal > Usufruct (ID 73)
Property > Servitudes > Praedial (ID 58)
Property > Servitudes > Praedial > Rights of Light (ID 72)
Property > Servitudes > Praedial > Rights of Water (ID 71)
Property > Servitudes > Praedial > Rights of Way (ID 70)
Public Law (ID 110)
Public Law > Administrative Law (ID 112)
Public Law > Administrative Law > Archive (ID 114)
Public Law > Administrative Law > Judiciary (ID 113)
Public Law > Administrative Law > Municipal Government  (ID 115)
Public Law > Administrative Law > Provincial Government  (ID 123)
Public Law > Taxation (ID 111)
Status (ID 1)
Status > Citizenship (ID 7)
Status > Citizenship > Captivity (ID 41)
Status > Citizenship > Freed Persons (ID 105)
Status > Citizenship > Naturalization (ID 40)
Status > Family (ID 5)
Status > Family > Divorce (ID 22)
Status > Family > Marital Arrangments (ID 23)
Status > Family > Marital Arrangments > Dowry (ID 24)
Status > Family > Marital Arrangments > Property of Women (ID 27)
Status > Family > Marital Arrangments > Provision for Wife (ID 25)
Status > Family > Marital Arrangments > Provisions for Children (ID 26)
Status > Family > Marriage (ID 21)
Status > Family > Marriage > Acts of Marriage (ID 117)
Status > Family > Marriage > Legitimate Children (ID 118)
Status > Inheritance (ID 6)
Status > Inheritance > Debts and Obligations (ID 119)
Status > Inheritance > Intestacy (ID 28)
Status > Inheritance > Intestacy > Classes of Heirs (ID 30)
Status > Inheritance > Intestacy > Father Inheritance Rights  (ID 103)
Status > Inheritance > Intestacy > Firstborn Inheritance Rights (ID 104)
Status > Inheritance > Intestacy > Women Inheritance rights (ID 31)
Status > Inheritance > Testaments (ID 29)
Status > Inheritance > Testaments > Accaptance of will (ID 35)
Status > Inheritance > Testaments > Capacity (ID 34)
Status > Inheritance > Testaments > Causa Mortis (ID 32)
Status > Inheritance > Testaments > Deeds of Gift as Wills (ID 37)
Status > Inheritance > Testaments > Disinheritance (ID 33)
Status > Inheritance > Testaments > Forms of Will (ID 36)
Status > Inheritance > Testaments > Legacies (ID 38)
Status > Inheritance > Testaments > Trust (ID 39)
Status > Legal Capacity (ID 9)
Status > Legal Capacity > Adoption (ID 47)
Status > Legal Capacity > Guardianship (ID 48)
Status > Legal Capacity > Guardianship > Guardianship over Minors (ID 53)
Status > Legal Capacity > Guardianship > Guardianship over Women (ID 54)
Status > Legal Capacity > Infamy (ID 51)
Status > Legal Capacity > Legal Capcity of Women (ID 120)
Status > Legal Capacity > Mental Disability (ID 50)
Status > Legal Capacity > Minors (ID 49)
Status > Legal Capacity > Paternal Power (ID 46)
Status > Legal Capacity > Paternal Power > Emanciaption (ID 52)
Status > Slavery (ID 8)
Status > Slavery > Manumission (ID 43)
Status > Slavery > Manumission > Paramone (ID 45)
Status > Slavery > Manumission > Pseudo Manumissons (ID 44)
Status > Slavery > Obligations of Slaves (ID 42)
Status > Slavery > Patronage/Clientela (ID 106)

## Output Format
Return ONLY a valid JSON array of objects. Do not include markdown formatting like ```json, preamble, or conversational text. Use the exact data types and structure shown in this example:

[
  {{
    "field": level_0 > level_1 > level_2,
    "field_id": 123
  }},
  {{
    "field": level_0,
    "field_id": 38
  }}
]

## Input Data
Source Text, {reference_name} ({language}):
{text}

English Translation:
{translation}




"""

INDEX_0_1 = """## Context
For a project comparing different legal cultures in the Roman world, you are tagging specific sources from each culture, in order to identify similarities and differences between them. The project includes three types of tags: judicial topics, keywords and index. You will be identifying the judicial topics of the sources. For any given source choose the most relevant judicial topics from the provided list of topics below.  

## Instructions
1. Analyze the provided {language} source.
 Interpret the legal meaning of the source, as closely as possible to the specific matters addressed in it.

2. Select from the provided source legal terminology, or specific words or short phrases that are essential to the specific legal matters, legal principles or legal situations discussed in the text. 

3. Exclude very common, high-frequency words and general vocabulary that are not pertinent to the legal matter at hand (such as particles, adjectives, adverbs, names, objects that are not relevant to the legal matter) 

4. For single words, return the dictionary form (lemma) of the selected term. For fixed phrases return its accepted form (e.g. “in potestate”) 

5. Guardrails:
Do not return words that don’t appear in the text of the provided source.
Return only words in the original language of the source.

## Examples
[1] Hebrew: Mishnah Bava Batra 8:5
האומר איש פלוני בני בכור לא יטל פי שנים, איש פלוני בני לא יירש עם אחיו, לא אמר כלום, שהתנה על מה שכתוב בתורה. המחלק נכסיו לבניו על פיו, רבה לאחד ומעט לאחד והשוה להן את הבכור, דבריו קימין. ואם אמר משום ירשה, לא אמר כלום. כתב בין בתחלה בין באמצע בין בסוף משום מתנה, דבריו קימין. האומר איש פלוני יירשני במקום שיש בת, בתי תירשני במקום שיש בן, לא אמר כלום, שהתנה על מה שכתוב בתורה. רבי יוחנן בן ברוקה אומר, אם אמר על מי שהוא ראוי לירשו, דבריו קימין. ועל מי שאין ראוי לירשו, אין דבריו קימין. הכותב את נכסיו לאחרים והניח את בניו, מה שעשה עשוי, אבל אין רוח חכמים נוחה הימנו. רבן שמעון בן גמליאל אומר, אם לא היו בניו נוהגין כשורה, זכור לטוב:

Index:
בכור, הכותב נכסיו, מחלק נכסיו, מתנה על מה שכתוב בתורה, ירושה, מתנה, נחלה, נכס, פי שנים, ראוי ליורשו, רוח חכמים נוחה הימנו
[2] Latin:  Ps. Ulpian 20.10
Filius familiae testamentum facere non potest, quoniam nihil suum habet, ut testari de eo possit. Sed divus Augustus Marcus constituit, ut filius familiae miles de eo peculio, quod in castris adquisivit, testamentum facere possit.

Index: 
castra, filius familiae, miles, peculium, testamentum, testari, testor

[3] Greek: Gnomon of Idios Logos 9
θ τ[ο]ὺς ἀπελευ[θ]έ̣ρους τῶν ἀστῶν \ἀτέκνους/ καὶ ἀδιαθέτους τελευτῶν τας κληρονο[μ]ο̣ῦσιν οἱ πάτρωνες ἢ οἱ τούτων υἱοί, ἐὰν ὦσικαὶ ἐπιδικά[ζ]ο̣νται, θυγατέρες δὲ ἢ ἄλλος τι\ς/ οὐ κληρονομήσουσι ἀλλὰ̣ ὁ φίσκος.

Index:
κληρονομέω, πάτρων, φίσκος, ἀδιάθετος, ἀπελεύθερος, ἀστός, ἄτεκνος, ἐπιδικάζομαι, ἐπιδικάζω 

## Output Format
Return ONLY a valid JSON array of strings. Do not include markdown formatting, headers, or conversational text.

Example format:
["term1", "term2", "term3"]

## Input Data
{context_section}Source Text, {source_name} ({language}):
{text}

"""

KEYWORDS_0_2_JTWC = '''## Context
For a project comparing different legal cultures in the Roman world, you are tagging specific sources from each culture, in order to identify similarities and differences between them. The project includes three types of tags: judicial topics, keywords and index. You will be identifying the keywords of the sources. For any given source choose the most relevant keywords from the provided List of Keywords below. 
To achieve the project’s goal, a high degree of correspondence between keywords across different sources is necessary. There is no point in adding a keyword that appears in only one source. The keywords should therefore be general enough to be applied to additional sources from different legal traditions.
 

## Detailed Instructions
1. Analyze the provided {language} legal source{translation_note}.
	a. Interpret the legal meaning of the source, as closely as possible to the specific matters addressed in it. Identify the situation, legal issues, legal terms, formulas, legal procedures, parties and significant objects in it.
	b. Take into consideration the source’s provided textual context, only if the meaning of the source is vague.  
2. Review the provided List of Keywords.
3. Identify and return the keywords from the List of Keywords that best match the situation, legal issues, legal terms, formulas, legal procedures, parties and significant objects in the source. 
*return keywords that may be clearly inferred from the source (by an expert in the field)
*refrain from returning redundant keywords that have a very similar meaning. Among similar keywords, return the one that is most relevant to the given source. .
4. If the List of Keywords does not include all relevant keywords, suggest new keywords. 
* All new keywords MUST be in English, unless it is a widely used latin term.
* refrain from suggesting redundant keywords that are synonymous with an existing keyword in the List of Keywords or that have a very similar meaning. We do not want to create too many words that will be too unique to specific texts.
* Assign any suggested keyword to the existing category that matches it best. If no existing category fits, use "other" for the category field.
* For suggested keywords, set the keyword_id to -1. Set the category_id to the ID of the matched category, or -1 if the category is "other".


## Examples
[1] Mishnah Bava Batra 8:7 
הכותב נכסיו לבניו, צריך שיכתב מהיום ולאחר מיתה, דברי רבי יהודה. רבי יוסי אומר, אינו צריך. הכותב נכסיו לבנו לאחר מותו, האב אינו יכול למכר, מפני שהן כתובין לבן, והבן אינו יכול למכר, מפני שהן ברשות האב. מכר האב, מכורין עד שימות. מכר הבן, אין ללוקח בהן כלום עד שימות האב. האב תולש ומאכיל לכל מי שירצה. ומה שהניח תלוש, הרי הוא של יורשין. 
Keywords: 
bequeathal clause, division of estate before death, father, son, from today and after death, gift after death, heir, inter vivos transfer, power of sale, property, usufructus, attached fruits, gathered fruits, temporary sale.  
[2] Ps. Ulpian 20.5
Ex duobus fratribus, qui in eiusdem patris potestate sunt, alter  familiae emptor, alter testis esse non potest, quoniam quod unus ex his mancipio accipit, adquirit patri, cui filius suus testis esse non debet.
Keywords: 
brother, buyer, disqualified witnesses, father, mancipatio, mancipatory will, paterfamilias, patria potestas, scale holder, son, testamentum per aes et libram, witnesses

## List of Keywords 
{keyword_list}

## Output Format
Return ONLY a valid JSON array of objects. Do not include markdown formatting like ```json, preamble, or conversational text. Use the following schema:

[
  {{
    "category": "Name of the category",
    "keyword": "Matched or suggested word",
    "suggested": true/false,
    "category_id": 123,
    "keyword_id": 456
  }}
]

## Input Data
Broader Context:
{broader_context}

Source Text, {source_name} ({language}):
{text}
{translation_section}
'''

KEYWORDS_0_2_PI = '''## Context
For a project comparing different legal cultures in the Roman world, you are tagging specific sources from each culture, in order to identify similarities and differences between them. The project includes three types of tags: judicial topics, keywords and index. You will be identifying the keywords of the sources. For any given source choose the most relevant keywords from the provided list of topics below. 
To achieve the project’s goal, a high degree of correspondence between keywords across different sources is necessary. There is no point in adding a keyword that appears in only one source. The keywords should therefore be general enough to be applied to additional sources from different legal traditions.

## Instructions
1. Analyze the provided {language} legal source. [translation Y/N]
Divide source into its main sections. Consider standard documentary structures. 
Interpret the legal meaning of each section, as closely as possible to the specific matters addressed. Identify the situation, legal issues, legal terms, formulas, legal procedures, parties (not personal names) and significant objects in it.
Consider the possible judicial reality within which the source functioned.
2. Review the provided List of Keywords.
3. Identify and return the keywords from the List of Keywords that best match the situation, legal issues, legal terms, formulas, legal procedures, parties and significant objects in the source. 
*return keywords that may be clearly inferred from the source (by an expert in the field)
*refrain from returning redundant keywords that have a very similar meaning. Among similar keywords, return the one that is most relevant to the given source. .
4. If the List of Keywords does not include all relevant keywords, suggest new keywords. 
* All new keywords MUST be in English, unless it is a widely used latin term..
* refrain from suggesting redundant keywords that are synonymous with an existing keyword in the List of Keywords or that have a very similar meaning. We do not want to create too many words that will be too unique to specific texts.
* Assign any suggested keyword to the existing category that matches it best. If no existing category fits, use "other" for the category field.
* For suggested keywords, set the keyword_id to -1. Set the category_id to the ID of the matched category, or -1 if the category is "other".


## Examples
[1] Egyptian papyri TM 20764
[ἔτους ̣ ̣ Αὐτοκράτορος Καίσαρος Τραιανοῦ Ἁδρια]νοῦ Σεβαστοῦ, Τῦβι ιγ, ἐν Ὀξυρύγχων πόλει τῆς Θηβαίδος, ἀγαθῇ τύχηι.
[τάδε διέθετο νοῶν καὶ φρονῶν Πεκῦσις Ἑρμοῦ τοῦ Π]εκύσιος μητρὸς Διδύμης τῆς Φιλώτου τῶν ἀπʼ Ὀξυρύγχων πόλεως ἐν ἀγυιᾷ· ἐφʼ ὃν μὲν περίειμι χρόνον ἔχειν με τὴν κατὰ τῶν ἐμῶν ἐξουσίαν
[- ca.37 - κ]α̣ὶ μεταδιατίθεσθαι. ἐὰν δὲ ἐπὶ ταύτῃ τελευτήσω τῇ διαθήκῃ, κληρονόμον ἀπολείπω τὴν θυγατέρα(*) μου Ἀμμωνοῦν μητρὸς Πτολεμᾶς, ἐὰν ζῇ, ε[ἰ δὲ]
[μή, τὴν ταύτης γενεάν, τῶν ὑπαρχόντων μοι] ἐπʼ ἀμφόδου Κρητικοῦ μερῶν κοινωνικῆς(*) οἰκίας καὶ αὐλῆς καὶ καμαρῶν. τὰ δὲ ὑπʼ ἐμοῦ ἀπολειφθησόμενα σκεύη καὶ ἔπιπλα καὶ ἐνδομενείαν καὶ εἴ τι ἄλλ[ο]
5[ἐὰν ἔχω, πάντα καταλείπω τῇ τῶν μὲν ἐμῶν τέκνω]ν μητρὶ ἐμοῦ δὲ γυναικὶ Πτολέμᾳ, ἀπελευθέρᾳ Δημητρίου Ἑρμίππου, ἐπὶ τῷ αὐτὴν ἔχειν ἐπὶ τὸν τῆς ζωῆς αὐτῆς χρόνον τὴν χρῆσιν καὶ ἐνοίκησιν καὶ ἐνοι-
[κίων ἀποφορὰν τῶν μερῶν](*)[ οἰκίας καὶ αὐλῆς καὶ καμ]αρῶν. ἐὰν δὲ συμβῇ τὴν Ἀμμωνοῦν ἄτεκνον καὶ ἀδιάθετον τελευτῆσαι, ἔσται τὰ μέρη τῶν ἐνγαίων τοῦ ὁμομητρίου αὐτῆς ἀδελφοῦ Ἀντᾶτος, ἐὰν ζῇ, εἰ δὲ μή,
[- ca.20 - καὶ μὴ ἐξεῖναι μηδενὶ ἄλλῳ π]α̣ρ̣ενχιρεῖν(*) τοῖς ὑπʼ ἐμοῦ διατεταγμένοις, ἢ τὸν παραβάντα τι τούτων ἀποτίνειν τῇ θυγατρὶ μου καὶ κληρονόμῳ Ἀμμωνοῦτι ἐπιτίμου δραχμὰς χειλίας(*) καὶ
[- ca.37 -] (hand 2) Πεκῦσις Ἑρμοῦ τοῦ Πεκύσιος καταλείπω μετὰ τελευτήν μου κληρονόμον τὴν θυγατέρα
[μου Ἀμμωνοῦν τῶν ἐπʼ ἀμφόδου Κρητι]κοῦ μερῶν οἰκίας καὶ αὐλῆς καὶ καμαρῶν· τῇ δὲ γυναικί μου Πτολέμᾳ καταλείπω πάν-
10[τα τὰ σκεύη μου καὶ ἔπιπλα καὶ ἐ]νδομενείαν καὶ εἴ τι ἄλλο αἰὰν(*) χω(*), καὶ ἐφʼ ὅσον ζῇ τὴν ἐνοίκησιν τῶν μερῶν τῆς οἰκ-
[ίας καὶ αὐλῆς καὶ καμαρῶν. ἐὰν δ]ὲ ἡ Ἀμμωνοῦς ἄτεκνος καὶ ἀδιάθετος τελευτήσῃ, ἔστω τὰ μέρη τῶν ἐνγαίων τοῦ
[ὁμομητρίου αὐτῆς ἀδελφοῦ Ἀ]ν[τ]ᾶτος ὡς πρόκιται. εἰμὶ ἐτῶν τεσσαράκοντα τεσσάρων , οὐλὴ τραχήλῳ ἐξ ἀριστερῶν,
[καὶ ἔστι μου ἡ σφραγὶς Ἄμ]μωνος.(*). (hand 3) Σαραπίων Σαραπίωνος τοῦ Διονυσίου ἀπὸ τῆς αὐτῆς πόλεως μαρτυρῶ τῇ τοῦ Πεκυσις(*) διαθήκῃ, καὶ
[εἰμὶ ἐτῶν ̣ ̣, οὐλὴ ̣ ̣ ̣ ̣ ̣ ̣, καὶ ἔστι μου ἡ σφ]ραγὶς Διονύσου. (hand 4) Ἑκάτων Σαραπίωνος τοῦ Ἑκάτωνος ἀπὸ τῆς αὐτῆς πόλεως μαρτυρῶ τῇ τοῦ Πεκύσιος διαθήκῃ, καὶ εἰμὶ
15[ἐτῶν ̣ ̣, οὐλὴ - ca.13 - , καὶ ἔστι μο]υ ἡ σφραγὶς Σαράπιδος. (hand 5) Παποντὼς Διογένους τοῦ Παποντῶτος ἀπὸ τῆς αὐτῆς πόλεως μαρτυρῶ τῇ τοῦ Πεκύσιος
[διαθήκῃ, καὶ εἰμὶ ἐτῶν ̣ ̣ ̣ ̣ ̣ ̣ ̣, καὶ] ἔστιν μου ἡ σφραγὶς Διὸς ἐπʼ ἀέτῳ(*). (hand 6) Ζωίλος Ζωίλου τοῦ Πανεχώτου τῶν ἀπὸ τῆς αὐτ-
[ῆς πόλεως μαρτυρῶ τῇ τοῦ Π]εκύσεος διαθήκῃ, καὶ ἰμὶ(*) ἐτῶν τεσσαράκοντα ὀκτὼ , \οὐλὴ/ πήχι(*) ἀριστερῷ, ἡ
[δὲ σφραγίς μού ἐστιν ̣ ̣ ̣ ̣ ̣ ̣ Ἁ]ρποκράτου ἐπὶ κιβωρτωι. (hand 7) Ἡρᾶς ὁ καὶ Σάιος Ζηνᾶτος τοῦ Ἡρᾶτος ἀπὸ τῆς αὐτῆς πόλεως μαρτυρῶι(*) τῇ τοῦ Πεκύσιος
[διαθήκῃ, καὶ εἰμὶ ἐτῶν - ca.10 -, οὐλὴ ἀντικνημ]ίωι δεξιῶι, καὶ ἔστι μου ἡ σφραγὶ[ς π]ρ[ο]τομὴ(*) φιλ[ο]σόφου. (hand 8) Διονύσιος Διον[υσ]ίου τ[ο]ῦ Διογένους ἀπὸ τῆς αὐτῆ[ς] πόλεως μαρτ[υ]ρῶ
20[τῇ τοῦ Πεκύσιος διαθήκῃ, καὶ εἰμὶ] ἐτῶν τεσσαράκοντα ἕξ , οὐλὴ παρὰ κρόταφον δεξιόν, καὶ ἔστι μου ἡ σφραγὶς Διονυσοπλάτωνος. (hand 9) μν̣η̣μ̣(ονείου)(*) Ὀξυρ(ύγχων) πόλ(εως).
[ἔτους ̣ ̣ Αὐτοκράτορος Καί]σαρος Τραιανοῦ Ἁδριανοῦ Σεβαστοῦ(*), Τῦβι ιγ.
[⁦ -ca.?- ⁩ διαθήκη Πεκύσιος Ἑρ]μοῦ τοῦ Πεκύσιος μητρὸ(ς) Διδύμης Φιλώτου ἀπʼ Ὀξ(υρύγχων) π[ό]λ(εως).
Translation: The ... year of the Emperor Trajanus Hadrianus Augustus, Tybi 13, at Oxyrhynchus in the Thebaid; for good luck. This is the will, made in the street, of Pekusis, son of Hermes and Didyme, daughter of Philotas, an inhabitant of Oxyrhynchus, being sane and in his right mind. So long as I survive, I am to have power over my property, to ... and to alter my will. But if I die with this will unchanged, I leave my daughter Ammonous whose mother is Ptolema, if she survive me, but if not, then her children, heir to my shares in the common house, court and rooms situated in the Cretan quarter. All the furniture, movables and household stock and other property whatsoever that I shall leave, I bequeath to the mother of my children and my wife, Ptolema, the freedwoman of Demetrius, son of Hermippus, with the condition that she shall have for her lifetime the right of using, dwelling in, and building in the said house, court and rooms. If Ammonous should die without children and intestate, the share of the fixtures shall belong to her half-brother on the mother’s side, Antas, if he survive, but if not, to.... No one shall violate the terms of this my will under pain of paying to my daughter and heir Ammonous a fine of 1000 drachmae and (to the treasury an equal sum?) 
Keywords: 
Right of habitation for the widow, testamentary heir, bequeathal clause, currency, daughter, fine, freedperson, heir, house in the city, immovable property, movable property, penalty clause, penalty for breach of will, property, specification of assets, substitutio vulgaris, ordinary substitution, testament formula, testator, usufructus, witnesses, woman
[2] Greek Inscription SEG 9.7 (Will of Ptolemy VIII Euergetes II)
Ἔτους πεντεκαιδεκάτου, μηνὸς Λώιου.Ἀγαθῆι τύχηι. Τάδε διέθετο βασιλεὺς
Πτολεμαῖος βασιλέως Πτολεμαίου
καὶ βασιλίσσης Κλεοπάτρας, θεῶν
Ἐπιφανῶν, ὁ νεώτερος· ὧν καὶ τὰ ἀντίγραφα
εἰς Ῥώμην ἐξαπέσταλται. — Εἴη μέν μοι
μετὰ τῆς τῶν θεῶν εὐμενείας μετελθεῖν
καταξίως τοὺς συστησαμένους ἐπί με
τὴν ἀνόσιον ἐπιβουλὴν καὶ προελομένους
μὴ μόνον ⟦oν⟧ τῆς βασιλείας, ἀλλὰ καὶ
τοῦ ζῆν στερῆσαί με· ἐὰν δέ τι συμβαίνηι
τῶν κατ’ ἄνθρωπον πρότερον ἢ διαδόχους
ἀπολιπεῖν τῆς βασιλείας, καταλείπω
Ῥωμαίοις τὴν καθήκουσάν μοι βασιλείαν,
οἷς ἀπ’ ἀρχῆς τήν τε φιλίαν καὶ τὴν
συμμαχίαν γνησίως συντετήρηκα·
τοῖς δ’ αὐτοῖς παρακατατίθεμαι τὰ πράγματα
συντηρεῖν, ἐνευχόμενος κατά τε τῶν θεῶν
πάντων καὶ τῆς ἑαυτῶν εὐδοξίας, ἐάν τινες
ἐπίωσιν ἢ ταῖς πόλεσιν ἢ τῆι χώραι, βοηθεῖν
κατὰ τὴν ψιλίαν καὶ συμμαχίαν τὴν ⟦πρὸς⟧
πρὸς ἀλλήλους ἡμῖν γενομένην καὶ τὸ
δίκαιον παντὶ σθένει.
— Μάρτυρας δὲ τούτων ποιοῦμαι Δία τε τὸν
Καπετώλιον καὶ τοὺς Μεγάλους θεοὺς
καὶ τὸν Ἥλιον καὶ τὸν Ἀρχηγέτην Ἀπόλλωνα,
παρ’ ὧι καὶ τὰ περὶ τούτων ἀνιέρωται γράμματα.
Τύχηι τῆι ἀγαθῆι
Keywords: 
king, testamentary heir, deposit, heir, oath clauses, property, public document, testament formula, conditional bequeathal, testamentary intent, witnesses, bequeathal of kingdom, political alliance, divine witnesses. 

## List of Keywords 
{keyword_list}

## Output Format
Return ONLY a valid JSON array of objects. Do not include markdown formatting like ```json, preamble, or conversational text. Use the following schema:

[
  {{
    "category": "Name of the category",
    "keyword": "Matched or suggested word",
    "suggested": true/false,
    "category_id": 123,
    "keyword_id": 456
  }}
]

## Input Data
Source Text, {reference_name} ({language}):
{text}

English Translation:
{translation}

'''
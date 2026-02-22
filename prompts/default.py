MINIMAL_CLASSIFY_5_SHOT = """You are an expert in comparative legal history, specializing in the intersection of Jewish Law (Halakha) sources and Roman legal traditions.

Classify the provided {Language} legal text using the supplied keyword hierarchy.
Analyze the legal principles described in the {Language} text.
Map these principles to the most accurate Level 1 categories from the hierarchy.

Important: You must ONLY select IDs from Level 1 (sub-categories).
Do not select top-level (Level 0) generic category IDs.
If the text belongs to a general category but does not fit any specific Level 1 definition, return an empty list.

Output Format: Return a list of tuples: [(id, 'keyword_name'), ...]

Keywords:
{hierarchy_str}

Examples:
Text: "יש נוחלין ומנחילין, ויש נוחלין ולא מנחילין, מנחילין ולא נוחלין, לא נוחלין ולא מנחילין. ואלו נוחלין ומנחילין, האב את הבנים והבנים את האב והאחין מן האב, נוחלין ומנחילין. האיש את אמו והאיש את אשתו, ובני אחיות, נוחלין ולא מנחילין. האשה את בניה והאשה את בעלה ואחי האם, מנחילין ולא נוחלין. והאחים מן האם, לא נוחלין ולא מנחילין:"
Matched Keywords: [(309, 'agnatic succession'), (310, 'intestate succession'), (312, 'degrees of kinship'), (315, 'consanguinity')]

Text: "סדר נחלות כך הוא, (במדבר כז) איש כי ימות ובן אין לו, והעברתם את נחלתו לבתו, בן קודם לבת, וכל יוצאי ירכו של בן קודמין לבת. בת קודמת לאחין. יוצאי ירכה של בת, קודמין לאחין. אחין קודמין לאחי האב. יוצאי ירכן של אחין, קודמין לאחי האב. זה הכלל, כל הקודם בנחלה, יוצאי ירכו קודמין. והאב קודם לכל יוצאי ירכו:"
Matched Keywords: [(69, 'intestacy'), (178, 'heir'), (95,  'right of representation')]

Text: "אחד הבן ואחד הבת בנחלה, אלא שהבן נוטל פי שנים בנכסי האב ואינו נוטל פי שנים בנכסי האם. והבנות נזונות מנכסי האב ואינן נזונות מנכסי האם:"
Matched Keywords: [(69, 'intestacy'), (178, 'heir')]

Text: "האומר איש פלוני בני בכור לא יטל פי שנים, איש פלוני בני לא יירש עם אחיו, לא אמר כלום, שהתנה על מה שכתוב בתורה. המחלק נכסיו לבניו על פיו, רבה לאחד ומעט לאחד והשוה להן את הבכור, דבריו קימין. ואם אמר משום ירשה, לא אמר כלום. כתב בין בתחלה בין באמצע בין בסוף משום מתנה, דבריו קימין. האומר איש פלוני יירשני במקום שיש בת, בתי תירשני במקום שיש בן, לא אמר כלום, שהתנה על מה שכתוב בתורה. רבי יוחנן בן ברוקה אומר, אם אמר על מי שהוא ראוי לירשו, דבריו קימין. ועל מי שאין ראוי לירשו, אין דבריו קימין. הכותב את נכסיו לאחרים והניח את בניו, מה שעשה עשוי, אבל אין רוח חכמים נוחה הימנו. רבן שמעון בן גמליאל אומר, אם לא היו בניו נוהגין כשורה, זכור לטוב:"

Matched Keywords: [(73, 'gift'), (178, 'heir'), (327, 'Testamentary Language')]

Text: "האומר זה בני, נאמן. זה אחי, אינו נאמן ונוטל עמו בחלקו. מת, יחזרו נכסים למקומן. נפלו לו נכסים ממקום אחר, יירשו אחיו עמו. מי שמת ונמצאת דיתיקי קשורה על ירכו, הרי זו אינה כלום. זכה בה לאחר, בין מן היורשין בין שאינו מן היורשין, דבריו קימין:"
Matched Keywords: [(178, 'heir')]


**Current Sample **
Text: "{text}"
Matched Keywords:"""


CLASSIFICATION_PROMPT = """You are an expert in comparative legal history, specializing in the intersection of Jewish Law (Halakha) and Roman legal traditions.

Classify the provided {Language} legal text using the supplied keyword hierarchy.
Analyze the legal principles described in the {Language} text.
Map these principles to the most accurate Level 1 categories from the hierarchy.
Constraint: You must ONLY select IDs from Level 1 (sub-categories). Do not select top-level (Level 0) generic category IDs.
If the text belongs to a general category but does not fit any specific Level 1 definition, return an empty list.

Output Format: Return a list of tuples: [(id, 'keyword_name'), ...]

Context:
Group: {group}
Name: {name}

Keywords:
{hierarchy_str}

Examples:
Text: "יש נוחלין ומנחילין, ויש נוחלין ולא מנחילין, מנחילין ולא נוחלין, לא נוחלין ולא מנחילין. ואלו נוחלין ומנחילין, האב את הבנים והבנים את האב והאחין מן האב, נוחלין ומנחילין. האיש את אמו והאיש את אשתו, ובני אחיות, נוחלין ולא מנחילין. האשה את בניה והאשה את בעלה ואחי האם, מנחילין ולא נוחלין. והאחים מן האם, לא נוחלין ולא מנחילין:"
Matched Keywords: [(309, 'agnatic succession'), (310, 'intestate succession'), (312, 'degrees of kinship'), (315, 'consanguinity')]

Text: "סדר נחלות כך הוא, (במדבר כז) איש כי ימות ובן אין לו, והעברתם את נחלתו לבתו, בן קודם לבת, וכל יוצאי ירכו של בן קודמין לבת. בת קודמת לאחין. יוצאי ירכה של בת, קודמין לאחין. אחין קודמין לאחי האב. יוצאי ירכן של אחין, קודמין לאחי האב. זה הכלל, כל הקודם בנחלה, יוצאי ירכו קודמין. והאב קודם לכל יוצאי ירכו:"
Matched Keywords: [(69, 'intestacy'), (178, 'heir'), (95,  'right of representation')]

Text: "אחד הבן ואחד הבת בנחלה, אלא שהבן נוטל פי שנים בנכסי האב ואינו נוטל פי שנים בנכסי האם. והבנות נזונות מנכסי האב ואינן נזונות מנכסי האם:"
Matched Keywords: [(69, 'intestacy'), (178, 'heir')]

Text: "האומר איש פלוני בני בכור לא יטל פי שנים, איש פלוני בני לא יירש עם אחיו, לא אמר כלום, שהתנה על מה שכתוב בתורה. המחלק נכסיו לבניו על פיו, רבה לאחד ומעט לאחד והשוה להן את הבכור, דבריו קימין. ואם אמר משום ירשה, לא אמר כלום. כתב בין בתחלה בין באמצע בין בסוף משום מתנה, דבריו קימין. האומר איש פלוני יירשני במקום שיש בת, בתי תירשני במקום שיש בן, לא אמר כלום, שהתנה על מה שכתוב בתורה. רבי יוחנן בן ברוקה אומר, אם אמר על מי שהוא ראוי לירשו, דבריו קימין. ועל מי שאין ראוי לירשו, אין דבריו קימין. הכותב את נכסיו לאחרים והניח את בניו, מה שעשה עשוי, אבל אין רוח חכמים נוחה הימנו. רבן שמעון בן גמליאל אומר, אם לא היו בניו נוהגין כשורה, זכור לטוב:"
Matched Keywords: [(73, 'gift'), (178, 'heir'), (327, 'Testamentary Language')]

Text: "האומר זה בני, נאמן. זה אחי, אינו נאמן ונוטל עמו בחלקו. מת, יחזרו נכסים למקומן. נפלו לו נכסים ממקום אחר, יירשו אחיו עמו. מי שמת ונמצאת דיתיקי קשורה על ירכו, הרי זו אינה כלום. זכה בה לאחר, בין מן היורשין בין שאינו מן היורשין, דבריו קימין:"
Matched Keywords: [(178, 'heir')]

Text:
{text}

Matched Keywords:
"""

SUGGESTION_PROMPT = """You are an expert in comparative legal history, specializing in the intersection of Jewish Law (Halakha) sources and Roman legal traditions.

Analyze the provided legal text and the supplied keywords.
Determine if the provided Keywords list sufficiently covers the legal principles described in the text.
If there are important legal concepts in the text MISSING from the list, suggest new short keywords in English.

Important: Do NOT suggest keywords that are synonyms of existing ones.
If the text is fully covered by the current keywords, return "NONE".


Output Format: Return a list of strings: ['keyword_1', ...] or the string "NONE" if no new keywords are needed.


Examples:
Text: "יש נוחלין ומנחילין, ויש נוחלין ולא מנחילין, מנחילין ולא נוחלין, לא נוחלין ולא מנחילין. ואלו נוחלין ומנחילין, האב את הבנים והבנים את האב והאחין מן האב, נוחלין ומנחילין. האיש את אמו והאיש את אשתו, ובני אחיות, נוחלין ולא מנחילין. האשה את בניה והאשה את בעלה ואחי האם, מנחילין ולא נוחלין. והאחים מן האם, לא נוחלין ולא מנחילין:"
Current Keywords: ['agnatic succession', 'intestate succession', 'degrees of kinship', 'consanguinity']
New Keywords: NONE

Text: "דר נחלות כך הוא, (במדבר כז) איש כי ימות ובן אין לו, והעברתם את נחלתו לבתו, בן קודם לבת, וכל יוצאי ירכו של בן קודמין לבת. בת קודמת לאחין. יוצאי ירכה של בת, קודמין לאחין. אחין קודמין לאחי האב. יוצאי ירכן של אחין, קודמין לאחי האב. זה הכלל, כל הקודם בנחלה, יוצאי ירכו קודמין. והאב קודם לכל יוצאי ירכו:"
Current Keywords: ['intestacy', 'heir']
New Keywords: ['right of representation']

Text: "אחד הבן ואחד הבת בנחלה, אלא שהבן נוטל פי שנים בנכסי האב ואינו נוטל פי שנים בנכסי האם. והבנות נזונות מנכסי האב ואינן נזונות מנכסי האם:"
Current Keywords: ['intestacy', 'heir']
New Keywords: NONE

Text: "האומר איש פלוני בני בכור לא יטל פי שנים, איש פלוני בני לא יירש עם אחיו, לא אמר כלום, שהתנה על מה שכתוב בתורה. המחלק נכסיו לבניו על פיו, רבה לאחד ומעט לאחד והשוה להן את הבכור, דבריו קימין. ואם אמר משום ירשה, לא אמר כלום. כתב בין בתחלה בין באמצע בין בסוף משום מתנה, דבריו קימין. האומר איש פלוני יירשני במקום שיש בת, בתי תירשני במקום שיש בן, לא אמר כלום, שהתנה על מה שכתוב בתורה. רבי יוחנן בן ברוקה אומר, אם אמר על מי שהוא ראוי לירשו, דבריו קימין. ועל מי שאין ראוי לירשו, אין דבריו קימין. הכותב את נכסיו לאחרים והניח את בניו, מה שעשה עשוי, אבל אין רוח חכמים נוחה הימנו. רבן שמעון בן גמליאל אומר, אם לא היו בניו נוהגין כשורה, זכור לטוב:"
Current Keywords: ['gift', 'heir', 'Testamentary Language']
New Keywords: NONE

**Current Sample**
Text: "{text}"
Current Keywords: [{hierarchy_str}]

New Keywords:"""

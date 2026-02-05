from typing import List

class KeywordManager:
    def __init__(self):
        self.new_keywords: List[str] = []

    def update_keywords(self, new_keywords_list: List[str]) -> List[str]:
        """
        Validates and adds new keywords.
        For now, simplistic implementation: just adds unique ones.
        Returns the list of actually added keywords (not duplicates).
        """
        added = []
        for kw in new_keywords_list:
            if kw and kw not in self.new_keywords:
                # Basic validation: ensure it's not empty
                # No lemmatization as per user request for baseline
                self.new_keywords.append(kw)
                added.append(kw)
        return added

    def get_all_new_keywords(self) -> List[str]:
        return self.new_keywords

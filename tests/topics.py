import csv
from dataclasses import dataclass
from typing import List


@dataclass
class Topic:
    id: int
    name: str
    parent_id: int
    indented_topics: str
    full_path: str
    level: int


def load_topics(filepath: str) -> List[Topic]:
    """Load topics from a CSV file into a list of Topic objects."""
    topics = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            topics.append(Topic(
                id=int(row['Id']),
                name=row['Topic'],
                parent_id=int(row['Parent Topic Id']),
                indented_topics=row['Indented Topics'],
                full_path=row['Full Path'],
                level=int(row['Level'])
            ))
    return topics


def format_judicial_fields(topics: List[Topic]) -> str:
    """
    Create a string of all judicial fields and their IDs, organized by hierarchy.
    """
    judicial_lines = []

    # Sort them by their full path to ensure alphabetical/hierarchical grouping
    topics.sort(key=lambda x: x.full_path)

    for t in topics:
        # Format: Full Path (ID X)
        judicial_lines.append(f"{t.full_path} (ID {t.id})")

    return "\n".join(judicial_lines)


# --- Example Usage ---
if __name__ == "__main__":
    # Adjust path as necessary
    topics_list = load_topics("../data/Topics.csv")
    formatted_output = format_judicial_fields(topics_list)

    print(formatted_output)
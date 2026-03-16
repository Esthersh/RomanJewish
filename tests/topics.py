import csv
from typing import List, Dict


class Topic:
    """A data class to hold Judicial Topic metadata."""

    def __init__(self, id: int, name: str, parent_id: int, level: int):
        self.id = id
        self.name = name
        self.parent_id = parent_id
        self.level = level


def load_topics(csv_path: str) -> List[Topic]:
    """
    Load topics from the CSV file into a list of Topic objects.
    """
    topics = []
    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            topics.append(Topic(
                id=int(row['Id']),
                name=row['Topic'].strip(),
                parent_id=int(row['Parent Topic Id']),
                level=int(row['Level'])
            ))
    return topics


def format_judicial_topics(topics: List[Topic]) -> str:
    """
    Create a string of all judicial topics and their IDs organized by hierarchy.
    Categories are level 0 topics. Sub-topics are levels 1, 2, 3, etc.
    Format:
        Category {category_name}, id: {category_id}
          - {topic_name} (id: {topic_id})
            - {sub_topic_name} (id: {sub_topic_id})
    """
    # Separate top-level categories (level 0)
    categories = [t for t in topics if t.level == 0]

    # Build mapping: parent_id -> list of children
    children_map: Dict[int, List[Topic]] = {}
    for t in topics:
        if t.level > 0:
            children_map.setdefault(t.parent_id, []).append(t)

    output = []

    # Recursive helper function to handle arbitrary depths (levels 1, 2, 3...)
    def _add_children(parent_id: int, indent_level: int):
        children = children_map.get(parent_id, [])
        for child in children:
            # Add 2 spaces of indentation per level
            indent = "  " * indent_level
            output.append(f"{indent}- {child.name} (id: {child.id})")

            # Recurse for deeper sub-topics
            _add_children(child.id, indent_level + 1)

    # Build the output array
    for cat in categories:
        output.append(f"Category {cat.name}, id: {cat.id}")
        _add_children(cat.id, 1)
        output.append("")  # blank line between top-level categories

    return "\n".join(output).rstrip()


if __name__ == "__main__":
    # Load the topics from the CSV file
    topics_list = load_topics("../data/Topics.csv")

    # Format and print the hierarchy
    formatted_hierarchy = format_judicial_topics(topics_list)
    print(formatted_hierarchy)
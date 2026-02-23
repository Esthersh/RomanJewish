import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

@dataclass
class Keyword:
    id: int
    name: str
    level: int
    parent_id: Optional[int]
    full_path: str
    indented_name: str

@dataclass
class CorpusSample:
    source_id: str
    source_name: str
    group: str
    name: str
    text: str
    language: str
    original_row: Dict[str, Any]

class DataLoader:
    def __init__(self):
        pass

    def load_keywords(self, csv_path: str) -> List[Keyword]:
        """
        Loads keywords from CSV. 
        Expected columns: Id, Keyword, Parent KW Id, Indented Keywords, Full Path, Level
        """
        try:
            df = pd.read_csv(csv_path)
            keywords = []
            for _, row in df.iterrows():
                # Handle potential NaN for Parent KW Id (root nodes)
                parent_id = int(row['Parent KW Id']) if pd.notna(row['Parent KW Id']) and row['Parent KW Id'] != 0 else None
                
                kw = Keyword(
                    id=int(row['Id']),
                    name=str(row['Keyword']).strip(),
                    level=int(row['Level']),
                    parent_id=parent_id,
                    full_path=str(row['Full Path']).strip(),
                    indented_name=str(row['Indented Keywords'])
                )
                keywords.append(kw)
            return keywords
        except Exception as e:
            print(f"Error loading keywords: {e}")
            return []

    def load_corpus(self, csv_path: str) -> List[CorpusSample]:
        """
        Loads corpus samples from CSV.
        """
        try:
            df = pd.read_csv(csv_path)
            # filter annotated rows
            df.dropna(subset="Keywords", inplace=True)
            
            # Forward fill Group and Name columns
            if 'Group' in df.columns:
                df['Group'] = df['Group'].ffill()
            if 'Name' in df.columns:
                df['Name'] = df['Name'].ffill()
                
            samples = []
            for _, row in df.iterrows():
                # Adjust column names based on actual file inspection if needed
                
                sample = CorpusSample(
                    source_id=str(row.get('SourceID', '')),
                    source_name=str(row.get('Refference', '')), # Using Refference as source name fallback/equivalent
                    group=str(row.get('Group', '')),
                    name=str(row.get('Name', '')),
                    text=str(row.get('Text', '')),
                    language=str(row.get('Language', 'Hebrew')), # Default to Hebrew if missing, as prompt example implies
                    original_row=row.to_dict()
                )
                samples.append(sample)
            return samples
        except Exception as e:
            print(f"Error loading corpus: {e}")
            return []

import sys

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
class JudicialField:
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
    text: str
    language: str
    original_row: Dict[str, Any]
    ref_id: float
    context_text: str = ''


class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_keywords(csv_path: str) -> List[Keyword]:
        """
        Loads keywords from CSV. 
        Expected columns: Id, Keyword, Parent KW Id, Indented Keywords, Full Path, Level
        """
        try:
            df = pd.read_csv(csv_path)  ## TODO: handle file not found
            keywords = []
            for _, row in df.iterrows():
                # Handle potential NaN for Parent KW Id (root nodes)
                parent_id = int(row['Parent KW Id']) if (pd.notna(row['Parent KW Id'])
                                                         and row['Parent KW Id'] != 0) else None

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
            sys.exit(1)

    @staticmethod
    def load_judicial_fields(csv_path: str) -> List[JudicialField]:
        """
        Loads judicial fields from Topics.csv.
        Expected columns: Id, Topic, Parent Topic Id, Indented Topics, Full Path, Level
        """
        try:
            df = pd.read_csv(csv_path)
            fields = []
            for _, row in df.iterrows():
                parent_id = int(row['Parent Topic Id']) if (pd.notna(row['Parent Topic Id'])
                                                            and row['Parent Topic Id'] != 0) else None

                field = JudicialField(
                    id=int(row['Id']),
                    name=str(row['Topic']).strip(),
                    level=int(row['Level']),
                    parent_id=parent_id,
                    full_path=str(row['Full Path']).strip(),
                    indented_name=str(row['Indented Topics'])
                )
                fields.append(field)
            return fields
        except Exception as e:
            print(f"Error loading judicial fields: {e}")
            sys.exit(1)

    @staticmethod
    def load_corpus(csv_path: str, include_non_english: bool = False, include_unannotated: bool = False,
                    analyzed_only: bool = False, context_filter: str = 'any') -> List[CorpusSample]:
        """
        Loads corpus samples from CSV.
        """
        try:
            df = pd.read_csv(csv_path)
            # filter rows that have an English translation and non-empty Keywords
            if not include_non_english:
                df = df.dropna(subset=["English"])
                if df['English'].dtype == object:
                    df = df[df['English'].str.strip() != '']
                    
            if not include_unannotated:
                df = df.dropna(subset=["Keywords"])
                # Also ensure they are not just blank strings if they are object type
                if df['Keywords'].dtype == object:
                    df = df[df['Keywords'].str.strip() != '']

            if analyzed_only:
                df = df[df['Analyzed [y/n]'].fillna('').str.strip().str.lower() == 'y']

            if context_filter == 'with':
                df = df[df['context text'].notna() & (df['context text'].astype(str).str.strip() != '')]
            elif context_filter == 'without':
                has_ctx = df['context text'].notna() & (df['context text'].astype(str).str.strip() != '')
                df = df[~has_ctx]

            # Forward fill Group and Name columns
            if 'Group' in df.columns:
                df['Group'] = df['Group'].ffill()
            if 'Name' in df.columns:
                df['Name'] = df['Name'].ffill()

            samples = []
            for _, row in df.iterrows():
                # Adjust column names based on actual file inspection if needed
                ctx = row.get('context text', '')
                ctx = str(ctx).strip() if pd.notna(ctx) else ''

                sample = CorpusSample(
                    source_id=str(row.get('SourceID', '')),
                    source_name=str(row.get('Refference', '')), 
                    group=str(row.get('Group', '')),
                    text=str(row.get('Text', '')),
                    language=str(row.get('Language', '')),
                    ref_id=float(row.get('ref Code', '')),
                    original_row=row.to_dict(),
                    context_text=ctx
                )
                sample.original_row['translation'] = row.get('English', '')
                samples.append(sample)
            return samples
        except Exception as e:
            print(f"Error loading corpus: {e}")
            sys.exit(1)

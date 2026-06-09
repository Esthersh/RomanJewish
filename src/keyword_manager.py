"""A keyword vocabulary that grows as sources are processed.

The vocabulary is seeded from ``data/Keywords.csv`` (curated ids 1-356). When the
model suggests a keyword that isn't already present, it is folded in with a new id
starting at :data:`NEW_ID_START` (10000). Any ``keyword_id >= NEW_ID_START`` therefore
marks a model-suggested keyword — that threshold is the "is this one suggested?" signal.

Matching is exact and case-insensitive (whitespace-collapsed): a suggestion whose name
already exists in the vocabulary (seed *or* previously added) reuses the existing id
instead of creating a duplicate. This is what lets later sources reuse the keywords
earlier sources introduced, instead of re-coining near-identical variants.

State is portable: :meth:`KeywordVocabulary.to_snapshot` / :meth:`from_snapshot` make a
sequential run resumable, and :meth:`save_augmented_csv` writes seed + additions back out
in the same schema as the seed CSV (the original CSV is never modified).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

NEW_ID_START = 10000
OTHER_CATEGORY_ID = -1
OTHER_CATEGORY_NAME = "other"


def _norm(name: object) -> str:
    """Normalized form used for exact, case-insensitive matching."""
    return " ".join(str(name).strip().lower().split())


@dataclass
class Keyword:
    id: int
    name: str
    category_id: int
    category_name: str
    suggested: bool  # True if added during a run (id >= NEW_ID_START)


class KeywordVocabulary:
    def __init__(self) -> None:
        self.categories: dict[int, str] = {}      # category_id -> category name (seed order)
        self.keywords: dict[int, Keyword] = {}     # keyword_id -> Keyword
        self._by_norm: dict[str, int] = {}         # normalized name -> keyword_id
        self._next_id: int = NEW_ID_START

    # ------------------------------------------------------------------ seed
    @classmethod
    def from_csv(cls, path: str | Path) -> "KeywordVocabulary":
        """Seed from data/Keywords.csv (Level 0 = category, Level 1 = keyword)."""
        v = cls()
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["Level"].strip() == "0":
                v.categories[int(r["Id"])] = r["Keyword"].strip()
        for r in rows:
            if r["Level"].strip() != "0":
                pid = int(r["Parent KW Id"])
                v._add(Keyword(
                    id=int(r["Id"]),
                    name=r["Keyword"].strip(),
                    category_id=pid,
                    category_name=v.categories.get(pid, OTHER_CATEGORY_NAME),
                    suggested=False,
                ))
        return v

    def _add(self, kw: Keyword) -> None:
        self.keywords[kw.id] = kw
        self._by_norm[_norm(kw.name)] = kw.id

    # -------------------------------------------------------------- lookup / growth
    def get_by_name(self, name: object) -> Optional[Keyword]:
        kid = self._by_norm.get(_norm(name))
        return self.keywords.get(kid) if kid is not None else None

    def resolve(
        self,
        name: object,
        category_name: object = None,
        category_id: object = None,
    ) -> tuple[Keyword, bool]:
        """Resolve a model-returned keyword *by name* to a vocabulary entry.

        Returns ``(keyword, is_new)``. If the name already exists (seed or previously
        added), its existing id is reused (``is_new=False``) regardless of what the model
        claimed. Otherwise a new id (>= NEW_ID_START) is minted and the keyword is added
        under the matched category, or "other" (-1) if no category matches.

        Ids reported by the model are intentionally ignored — they are unreliable
        (we have seen them come back as strings, or as -1 for every suggestion).
        """
        existing = self.get_by_name(name)
        if existing is not None:
            return existing, False
        cid, cname = self._resolve_category(category_name, category_id)
        kw = Keyword(
            id=self._next_id,
            name=str(name).strip(),
            category_id=cid,
            category_name=cname,
            suggested=True,
        )
        self._next_id += 1
        self._add(kw)
        return kw, True

    def _resolve_category(self, name: object, cid: object) -> tuple[int, str]:
        """Map a model-provided category (by id, then by name) to a known category."""
        if cid is not None:
            try:
                cid_int = int(cid)
            except (TypeError, ValueError):
                cid_int = None
            if cid_int in self.categories:
                return cid_int, self.categories[cid_int]
        if name:
            target = _norm(name)
            for c_id, c_name in self.categories.items():
                if _norm(c_name) == target:
                    return c_id, c_name
        return OTHER_CATEGORY_ID, OTHER_CATEGORY_NAME

    # ---------------------------------------------------------------- rendering
    def render_list(self) -> str:
        """Render the current vocabulary as the prompt's "List of Keywords" block.

        Categories appear in seed order (any "other" bucket last); keywords are sorted
        case-insensitively within each category. Newly suggested keywords are rendered
        identically to seed ones so the model treats them as first-class options.
        """
        by_cat: dict[int, list[Keyword]] = {}
        for kw in self.keywords.values():
            by_cat.setdefault(kw.category_id, []).append(kw)

        order = list(self.categories.keys())
        if OTHER_CATEGORY_ID in by_cat and OTHER_CATEGORY_ID not in order:
            order.append(OTHER_CATEGORY_ID)

        blocks: list[str] = []
        for cid in order:
            kws = by_cat.get(cid)
            if not kws:
                continue
            cname = self.categories.get(cid, OTHER_CATEGORY_NAME)
            lines = [f"Category: {cname} (id: {cid})"]
            for kw in sorted(kws, key=lambda k: _norm(k.name)):
                lines.append(f"  - {kw.name} (id: {kw.id})")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    # -------------------------------------------------------------- persistence
    def added(self) -> list[Keyword]:
        """Keywords introduced during the run (id >= NEW_ID_START), in id order."""
        return sorted((k for k in self.keywords.values() if k.suggested), key=lambda k: k.id)

    def save_augmented_csv(self, path: str | Path) -> None:
        """Write seed + added keywords to a CSV in the same schema as the seed."""
        rows = []
        for cid, cname in self.categories.items():
            rows.append({
                "Id": cid, "Keyword": cname, "Parent KW Id": 0,
                "Indented Keywords": cname, "Full Path": cname, "Level": 0, "Suggested": "",
            })
        for kw in sorted(self.keywords.values(), key=lambda k: (k.category_id, _norm(k.name))):
            rows.append({
                "Id": kw.id,
                "Keyword": kw.name,
                "Parent KW Id": kw.category_id,
                "Indented Keywords": f"    {kw.name}",
                "Full Path": f"{kw.category_name} > {kw.name}",
                "Level": 1,
                "Suggested": "yes" if kw.suggested else "",
            })
        fields = ["Id", "Keyword", "Parent KW Id", "Indented Keywords", "Full Path", "Level", "Suggested"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    def to_snapshot(self) -> dict:
        """Serializable state for resuming a sequential run."""
        return {
            "next_id": self._next_id,
            "categories": {str(cid): name for cid, name in self.categories.items()},
            "keywords": [
                {"id": k.id, "name": k.name, "category_id": k.category_id,
                 "category_name": k.category_name, "suggested": k.suggested}
                for k in self.keywords.values()
            ],
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "KeywordVocabulary":
        v = cls()
        v._next_id = int(data["next_id"])
        v.categories = {int(cid): name for cid, name in data["categories"].items()}
        for k in data["keywords"]:
            v._add(Keyword(
                id=int(k["id"]), name=k["name"], category_id=int(k["category_id"]),
                category_name=k["category_name"], suggested=bool(k["suggested"]),
            ))
        return v

    def save_snapshot(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_snapshot(cls, path: str | Path) -> "KeywordVocabulary":
        return cls.from_snapshot(json.loads(Path(path).read_text(encoding="utf-8")))

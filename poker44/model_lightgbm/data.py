"""Public benchmark API client with local caching (self-contained copy)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import requests

DEFAULT_BASE_URL = "https://api.poker44.net/api/v1/benchmark"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmark_cache"


@dataclass
class BatchExample:
    """One training example: a batch of hands and its bot/human label."""

    hands: List[dict]
    label: int
    source_date: str
    release_version: str
    schema_version: str
    chunk_hash: str
    batch_index: int
    split: Optional[str] = None

    @property
    def example_id(self) -> str:
        return f"{self.chunk_hash}:{self.batch_index}"


@dataclass
class BenchmarkClient:
    """Thin client over the public benchmark API with disk caching."""

    base_url: str = DEFAULT_BASE_URL
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    timeout: int = 60
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, path: str = "", params: Optional[dict] = None) -> dict:
        url = self.base_url if not path else f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(4):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
                return payload.get("data", payload)
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("unreachable")

    def status(self) -> dict:
        return self._get()

    def latest_source_date(self) -> str:
        return self.status()["latestSourceDate"]

    def releases(self, limit: int = 30, before: Optional[str] = None) -> List[dict]:
        params = {"limit": limit}
        if before:
            params["before"] = before
        data = self._get("releases", params=params)
        if isinstance(data, dict):
            return data.get("releases", data.get("items", []))
        return data

    @property
    def _hash_dir(self) -> Path:
        d = self.cache_dir / "by_hash"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _index_path(self, source_date: str, split: Optional[str]) -> Path:
        name = source_date if not split else f"{source_date}_{split}"
        return self.cache_dir / f"index_{name}.json"

    def _hash_path(self, chunk_hash: str) -> Path:
        safe = "".join(c for c in chunk_hash if c.isalnum() or c in "-_")
        return self._hash_dir / f"{safe}.json"

    def _load_from_index(self, index_path: Path) -> Optional[List[dict]]:
        if not index_path.exists():
            return None
        hashes = json.loads(index_path.read_text(encoding="utf-8"))
        chunks: List[dict] = []
        for h in hashes:
            path = self._hash_path(h)
            if not path.exists():
                return None
            chunks.append(json.loads(path.read_text(encoding="utf-8")))
        return chunks

    def fetch_chunks(
        self,
        source_date: str,
        *,
        split: Optional[str] = None,
        limit: int = 24,
        use_cache: bool = True,
    ) -> List[dict]:
        index_path = self._index_path(source_date, split)
        if use_cache:
            cached = self._load_from_index(index_path)
            if cached is not None:
                return cached

        all_chunks: List[dict] = []
        hashes: List[str] = []
        cursor: Optional[str] = None
        while True:
            params = {"sourceDate": source_date, "limit": limit}
            if split:
                params["split"] = split
            if cursor:
                params["cursor"] = cursor
            data = self._get("chunks", params=params)
            for chunk in data.get("chunks", []):
                chunk_hash = chunk.get("chunkHash") or chunk.get("chunkId") or ""
                if chunk_hash:
                    self._hash_path(chunk_hash).write_text(
                        json.dumps(chunk), encoding="utf-8"
                    )
                    hashes.append(chunk_hash)
                all_chunks.append(chunk)
            cursor = data.get("nextCursor")
            if not cursor:
                break

        index_path.write_text(json.dumps(hashes), encoding="utf-8")
        return all_chunks

    @staticmethod
    def iter_examples(chunk_objects: Iterable[dict]) -> Iterator[BatchExample]:
        for chunk in chunk_objects:
            batches = chunk.get("chunks") or []
            labels = chunk.get("groundTruth")
            if labels is None:
                continue
            if len(batches) != len(labels):
                continue
            for idx, (batch, label) in enumerate(zip(batches, labels)):
                if batch is None:
                    continue
                yield BatchExample(
                    hands=list(batch),
                    label=int(label),
                    source_date=chunk.get("sourceDate", ""),
                    release_version=chunk.get("releaseVersion", ""),
                    schema_version=chunk.get("schemaVersion", ""),
                    chunk_hash=chunk.get("chunkHash", ""),
                    batch_index=idx,
                    split=chunk.get("split"),
                )

    def load_examples(
        self,
        source_dates: Iterable[str],
        *,
        split: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[BatchExample]:
        examples: List[BatchExample] = []
        for date in source_dates:
            chunk_objects = self.fetch_chunks(date, split=split, use_cache=use_cache)
            examples.extend(self.iter_examples(chunk_objects))
        return examples

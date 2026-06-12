"""
TitanAI — RAG Manager (BM25)
=============================
Indexes all JSONL training data on startup.
At query time, retrieves top-k relevant Q&A pairs and injects
them as context into the prompt BEFORE generation.

Zero extra GPU usage — pure Python BM25 keyword search.
"""
import json
import logging
import re
import threading
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("titan.rag")

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    log.warning("[RAG] rank_bm25 not installed — RAG disabled. pip install rank_bm25")


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s_\-\.]", " ", text)
    return text.split()


class RAGManager:
    """
    Singleton RAG manager. Builds a BM25 index over all assistant
    Q&A pairs found in the JSONL training data.
    """

    _instance: Optional["RAGManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._docs: List[dict] = []       # {"q": str, "a": str, "source": str}
        self._corpus_tokens: List[List[str]] = []
        self._bm25: Optional["BM25Okapi"] = None
        self._indexed = False
        self._index_lock = threading.Lock()

    @classmethod
    def get(cls) -> "RAGManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Indexing ──────────────────────────────────────────────────────────────

    def build_index(self, data_root: Optional[str] = None) -> int:
        """
        Scan all .jsonl files under data_root (default: workspace data dir).
        Index every assistant message paired with its preceding user message.
        Returns number of documents indexed.
        """
        if not _BM25_AVAILABLE:
            return 0

        base = Path(data_root or Path(__file__).parent.parent.parent / "data")
        jsonl_files = list(base.rglob("*.jsonl"))

        if not jsonl_files:
            log.warning(f"[RAG] No JSONL files found under {base}")
            return 0

        docs = []
        for jf in jsonl_files:
            # skip the dpo feedback file itself
            if "dpo_feedback" in str(jf):
                continue
            try:
                with open(jf, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        messages = obj.get("messages", [])
                        # Find user → assistant pairs
                        for i, msg in enumerate(messages):
                            if msg.get("role") == "user" and i + 1 < len(messages):
                                nxt = messages[i + 1]
                                if nxt.get("role") == "assistant":
                                    q = msg.get("content", "").strip()
                                    a = nxt.get("content", "").strip()
                                    if q and a and len(a) > 50:
                                        docs.append({
                                            "q": q,
                                            "a": a,
                                            "source": jf.name,
                                        })
            except Exception as e:
                log.debug(f"[RAG] Skipping {jf.name}: {e}")

        if not docs:
            log.warning("[RAG] No Q&A pairs found in JSONL data.")
            return 0

        with self._index_lock:
            self._docs = docs
            # Tokenize the questions for BM25
            self._corpus_tokens = [_tokenize(d["q"]) for d in docs]
            self._bm25 = BM25Okapi(self._corpus_tokens)
            self._indexed = True

        log.info(f"[RAG] Indexed {len(docs):,} Q&A pairs from {len(jsonl_files)} JSONL files.")
        return len(docs)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.5) -> List[dict]:
        """
        Retrieve top_k most relevant Q&A pairs for query.
        Returns list of {"q": ..., "a": ..., "source": ..., "score": ...}
        """
        if not self._indexed or self._bm25 is None:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        with self._index_lock:
            scores = self._bm25.get_scores(tokens)

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        results = []
        for idx, score in ranked[:top_k]:
            if score < min_score:
                break
            results.append({**self._docs[idx], "score": round(float(score), 3)})

        return results

    # ── Context builder ───────────────────────────────────────────────────────

    def build_context_block(self, query: str, top_k: int = 3) -> str:
        """
        Returns a formatted context string to inject into the prompt,
        or empty string if nothing relevant found.
        """
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return ""

        lines = ["[Knowledge Context — retrieved from training corpus]"]
        for i, hit in enumerate(hits, 1):
            lines.append(f"\n--- Ref {i} (score={hit['score']}, src={hit['source']}) ---")
            lines.append(f"Q: {hit['q'][:300]}")
            lines.append(f"A: {hit['a'][:800]}")
        lines.append("[End Context]")
        return "\n".join(lines)

    @property
    def is_ready(self) -> bool:
        return self._indexed

    @property
    def doc_count(self) -> int:
        return len(self._docs)


# Global singleton
rag = RAGManager.get()

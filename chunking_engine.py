"""
Vast Multi-Strategy Chunking Engine for MSMARCO-XI RAG System

Implements:
1. Fixed-Size Chunking with Overlap
2. Semantic Boundary Chunking
3. Hierarchical Parent-Child Chunking
4. Metadata-Aware Window Chunking
"""

import re
import time
from typing import List, Dict, Any, Tuple
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, SEMANTIC_SIMILARITY_THRESHOLD

class Chunk:
    def __init__(self, text: str, chunk_id: str, doc_id: str, metadata: Dict[str, Any], parent_text: str = ""):
        self.text = text
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.metadata = metadata
        self.parent_text = parent_text or text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "parent_text": self.parent_text,
            "metadata": self.metadata
        }

class BaseChunker:
    def chunk_document(self, doc_id: str, text: str, doc_metadata: Dict[str, Any]) -> List[Chunk]:
        raise NotImplementedError

class FixedSizeOverlapChunker(BaseChunker):
    """
    Fixed character/token window chunker with overlapping boundaries.
    """
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, doc_id: str, text: str, doc_metadata: Dict[str, Any]) -> List[Chunk]:
        chunks = []
        start = 0
        text_len = len(text)
        idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end]
            meta = doc_metadata.copy()
            meta.update({
                "strategy": "fixed_overlap",
                "chunk_size": len(chunk_text),
                "start_char": start,
                "end_char": end
            })
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_fix_{idx}",
                doc_id=doc_id,
                metadata=meta
            ))
            idx += 1
            if end >= text_len:
                break
            start += (self.chunk_size - self.overlap)
        return chunks

class SemanticBoundaryChunker(BaseChunker):
    """
    Splits text on sentence boundaries and groups sentences into semantic blocks.
    """
    def __init__(self, max_chunk_chars: int = 300):
        self.max_chunk_chars = max_chunk_chars

    def chunk_document(self, doc_id: str, text: str, doc_metadata: Dict[str, Any]) -> List[Chunk]:
        # Sentence splitting pattern supporting English & Indic punctuation (।, ?, !, .)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?।\n])\s+', text) if s.strip()]
        if not sentences:
            sentences = [text]

        chunks = []
        current_chunk = []
        current_length = 0
        idx = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_length + sent_len > self.max_chunk_chars and current_chunk:
                chunk_text = " ".join(current_chunk)
                meta = doc_metadata.copy()
                meta.update({
                    "strategy": "semantic_boundary",
                    "sentence_count": len(current_chunk),
                    "chunk_size": len(chunk_text)
                })
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=f"{doc_id}_sem_{idx}",
                    doc_id=doc_id,
                    metadata=meta
                ))
                idx += 1
                current_chunk = []
                current_length = 0

            current_chunk.append(sent)
            current_length += sent_len

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            meta = doc_metadata.copy()
            meta.update({
                "strategy": "semantic_boundary",
                "sentence_count": len(current_chunk),
                "chunk_size": len(chunk_text)
            })
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_sem_{idx}",
                doc_id=doc_id,
                metadata=meta
            ))

        return chunks

class HierarchicalParentChildChunker(BaseChunker):
    """
    Generates small child chunks for precise vector retrieval while storing full parent context.
    """
    def __init__(self, child_size: int = 100):
        self.child_size = child_size

    def chunk_document(self, doc_id: str, text: str, doc_metadata: Dict[str, Any]) -> List[Chunk]:
        parent_text = text
        chunks = []
        start = 0
        text_len = len(text)
        idx = 0

        while start < text_len:
            end = min(start + self.child_size, text_len)
            child_text = text[start:end]
            meta = doc_metadata.copy()
            meta.update({
                "strategy": "parent_child",
                "is_child": True,
                "parent_len": len(parent_text)
            })
            chunks.append(Chunk(
                text=child_text,
                chunk_id=f"{doc_id}_child_{idx}",
                doc_id=doc_id,
                metadata=meta,
                parent_text=parent_text
            ))
            idx += 1
            if end >= text_len:
                break
            start += self.child_size

        return chunks

class MetadataAwareWindowChunker(BaseChunker):
    """
    Enriches chunks with document header metadata (Language, Query Topic, Passage Index)
    to maximize vector retrieval precision and semantic context grounding.
    """
    def __init__(self, chunk_size: int = 200):
        self.chunk_size = chunk_size

    def chunk_document(self, doc_id: str, text: str, doc_metadata: Dict[str, Any]) -> List[Chunk]:
        lang = doc_metadata.get("lang", "en")
        topic = doc_metadata.get("topic", "General")
        header = f"[Lang: {lang} | Topic: {topic} | Doc: {doc_id}]\n"

        chunks = []
        start = 0
        text_len = len(text)
        idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            sub_text = text[start:end]
            enriched_text = header + sub_text
            meta = doc_metadata.copy()
            meta.update({
                "strategy": "metadata_aware",
                "header_injected": True,
                "position_ratio": round(start / max(1, text_len), 2)
            })
            chunks.append(Chunk(
                text=enriched_text,
                chunk_id=f"{doc_id}_meta_{idx}",
                doc_id=doc_id,
                metadata=meta
            ))
            idx += 1
            if end >= text_len:
                break
            start += self.chunk_size

        return chunks

class MultiStrategyChunkingEngine:
    def __init__(self, strategy_name: str = "fixed_overlap"):
        self.strategy_name = strategy_name
        self.strategies: Dict[str, BaseChunker] = {
            "fixed_overlap": FixedSizeOverlapChunker(),
            "semantic_boundary": SemanticBoundaryChunker(),
            "parent_child": HierarchicalParentChildChunker(),
            "metadata_aware": MetadataAwareWindowChunker()
        }

    def set_strategy(self, strategy_name: str):
        if strategy_name in self.strategies:
            self.strategy_name = strategy_name

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        chunker = self.strategies.get(self.strategy_name, self.strategies["fixed_overlap"])
        all_chunks = []

        for doc in documents:
            doc_id = doc.get("id", "doc_0")
            passages = doc.get("passages", [])
            doc_meta = {"lang": doc.get("lang", "en"), "query": doc.get("query", "")}

            for p_idx, p in enumerate(passages):
                p_text = p.get("passage_text", "") if isinstance(p, dict) else str(p)
                p_meta = doc_meta.copy()
                p_meta["passage_idx"] = p_idx
                p_meta["is_selected"] = p.get("is_selected", 0) if isinstance(p, dict) else 0
                
                chunks = chunker.chunk_document(f"{doc_id}_p{p_idx}", p_text, p_meta)
                all_chunks.extend(chunks)

        return all_chunks

    def compare_strategies(self, documents: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Runs benchmarking across all 4 chunking strategies.
        """
        results = {}
        for name, chunker in self.strategies.items():
            t0 = time.perf_counter()
            chunks = []
            for doc in documents:
                doc_id = doc.get("id", "doc_0")
                passages = doc.get("passages", [])
                doc_meta = {"lang": doc.get("lang", "en")}
                for p_idx, p in enumerate(passages):
                    p_text = p.get("passage_text", "") if isinstance(p, dict) else str(p)
                    chunks.extend(chunker.chunk_document(f"{doc_id}_p{p_idx}", p_text, doc_meta))
            
            t_ms = (time.perf_counter() - t0) * 1000.0
            avg_len = sum(len(c.text) for c in chunks) / max(1, len(chunks))
            results[name] = {
                "chunk_count": len(chunks),
                "avg_chunk_chars": round(avg_len, 1),
                "chunking_time_ms": round(t_ms, 2),
                "metadata_richness": "High" if "meta" in name or "parent" in name else "Standard"
            }
        return results

if __name__ == "__main__":
    from dataset_loader import MSMARCOXIBackendLoader
    loader = MSMARCOXIBackendLoader()
    docs = loader.load_dataset()
    engine = MultiStrategyChunkingEngine()
    print("Benchmark results across chunking strategies:")
    print(engine.compare_strategies(docs[:10]))

"""
Dataset Loader for AI4Bharat MSMARCO-XI Multilingual RAG
Provides offline pre-bundled & cached multilingual knowledge base covering CSE subjects,
APIs, Databases, Operating Systems, Networks, DSA, AI, Cloud, Science, Economics, and Indic Knowledge.
"""

import json
from typing import List, Dict, Any
from pathlib import Path
from config import DATASET_NAME, DEFAULT_LANGUAGE, MAX_DATASET_SAMPLES, DATA_DIR, ALLOW_SYNTHETIC_DATA_FALLBACK

class MSMARCOXIBackendLoader:
    def __init__(self, lang: str = DEFAULT_LANGUAGE, max_samples: int = MAX_DATASET_SAMPLES):
        self.lang = lang
        self.max_samples = max_samples
        self.cache_file = DATA_DIR / f"msmarco_xi_{lang}.json"

    def load_dataset(self) -> List[Dict[str, Any]]:
        # 1. Load from static pre-bundled / cached index file (Fast & Reliable for Serverless / Offline)
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data:
                        return data[:self.max_samples]
            except Exception as e:
                print(f"[DatasetLoader] Notice loading cache {self.cache_file}: {e}")

        # 2. Optional online load from HuggingFace
        records = []
        try:
            from datasets import load_dataset
            ds = load_dataset(DATASET_NAME, self.lang, split="train", streaming=True)
            for idx, item in enumerate(ds):
                if idx >= self.max_samples:
                    break
                records.append({
                    "id": f"{self.lang}_{idx}",
                    "query": item.get("query", ""),
                    "answers": item.get("answers", []),
                    "passages": item.get("passages", []),
                    "lang": self.lang,
                    "source": "HuggingFace ai4bharat/MSMARCO-XI"
                })
        except Exception:
            pass

        if records:
            try:
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        return records[:self.max_samples]

if __name__ == "__main__":
    loader_en = MSMARCOXIBackendLoader(lang="en")
    data_en = loader_en.load_dataset()
    print(f"Loaded {len(data_en)} EN items successfully.")
    loader_hi = MSMARCOXIBackendLoader(lang="hi")
    data_hi = loader_hi.load_dataset()
    print(f"Loaded {len(data_hi)} HI items successfully.")

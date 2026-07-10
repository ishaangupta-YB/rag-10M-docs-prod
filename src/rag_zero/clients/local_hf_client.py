"""Local HuggingFace-based embedder / reranker / verifier loaders."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from rag_zero.clients.base import (
    BaseEmbedderClient,
    BaseRerankerClient,
    BaseVerifierClient,
)


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


class LocalEmbedderClient(BaseEmbedderClient):
    """Sentence-transformers style local embedder."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        query_instruction: str = "",
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.query_instruction = query_instruction
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def _mean_pool(
        self, last_hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        masked = last_hidden * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    async def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if is_query and self.query_instruction:
            texts = [self.query_instruction + t for t in texts]

        all_embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                embeddings = _normalize(pooled.cpu().numpy())
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, self.dim))

    @property
    def dim(self) -> int:
        return int(self.model.config.hidden_size)

    async def encode_query(self, query: str) -> np.ndarray:
        encoded = await self.encode([query], is_query=True)
        return cast("np.ndarray", encoded[0])


class LocalRerankerClient(BaseRerankerClient):
    """Cross-encoder reranker that returns scores via sigmoid over logits."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()

    async def score(self, query: str, documents: list[str]) -> list[float]:
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i : i + self.batch_size]
                encoded = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                logits = outputs.logits.squeeze(-1).cpu().float().numpy()
                # Cross-encoders typically produce a single logit per pair.
                batch_scores = 1.0 / (1.0 + np.exp(-logits))
                if batch_scores.ndim == 0:
                    batch_scores = np.array([float(batch_scores)])
                scores.extend([float(s) for s in batch_scores.tolist()])
        return scores


class LocalVerifierClient(BaseVerifierClient):
    """Local NLI-style verifier (entailment = supported)."""

    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.id2label = getattr(self.model.config, "id2label", None) or {
            0: "entailment",
            1: "neutral",
            2: "contradiction",
        }

    async def verify(self, claim: str, evidence: list[str]) -> float:
        if not evidence:
            return 0.0
        context = "\n".join(evidence)
        if not claim.strip():
            return 0.0
        encoded = self.tokenizer(
            context,
            claim,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits[0].cpu().float()
        probs = torch.softmax(logits, dim=-1).numpy()
        # Map label ids to canonical names.
        labels = {str(v).lower(): k for k, v in self.id2label.items()}
        entailment_idx = labels.get("entailment")
        contradiction_idx = labels.get("contradiction")
        if entailment_idx is None or contradiction_idx is None:
            entailment_idx = 0
            contradiction_idx = 2
        entailment = float(probs[entailment_idx])
        contradiction = float(probs[contradiction_idx])
        # Return support score with a small penalty for contradiction.
        return max(0.0, entailment - 0.1 * contradiction)

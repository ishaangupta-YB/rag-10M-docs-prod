"""Synthetic vector benchmark for LanceDB IVF_PQ at 100k, 1M, 10M scale."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import lancedb
import numpy as np


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


class ScaleLab:
    """Build, query, and project LanceDB IVF_PQ performance.

    Insertion is streamed in batches so that 10M 1024-dim vectors can be
    benchmarked without materialising the entire dataset in Python lists.
    """

    DEFAULT_BATCH_SIZE = 100_000

    def __init__(
        self,
        uri: Path | str,
        dim: int = 1024,
        num_queries: int = 100,
        seed: int = 42,
    ) -> None:
        self.uri = Path(uri)
        self.dim = dim
        self.num_queries = num_queries
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _generate_vectors(self, n: int) -> np.ndarray:
        vectors = self.rng.normal(size=(n, self.dim)).astype(np.float32)
        return _normalize(vectors)

    def _make_table(self, uri: Path, n: int, table_name: str) -> lancedb.Table:
        if uri.exists():
            shutil.rmtree(uri, ignore_errors=True)
        uri.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(uri))

        batch_size = min(self.DEFAULT_BATCH_SIZE, n)
        vectors = self._generate_vectors(batch_size)
        records = [{"id": i, "vector": vectors[i].tolist()} for i in range(batch_size)]
        table = db.create_table(table_name, data=records)

        for offset in range(batch_size, n, batch_size):
            remaining = n - offset
            current_batch = min(self.DEFAULT_BATCH_SIZE, remaining)
            vectors = self._generate_vectors(current_batch)
            records = [
                {"id": offset + i, "vector": vectors[i].tolist()}
                for i in range(current_batch)
            ]
            table.add(records)
        return table

    def _create_index(self, table: lancedb.Table, n: int) -> None:
        if n < 10_000:
            return
        num_partitions = max(32, min(4096, n // 10_000))
        candidates = [s for s in (8, 16, 32, 64, 128) if self.dim % s == 0]
        num_sub_vectors = candidates[-1] if candidates else self.dim
        try:
            table.create_index(
                vector_column_name="vector",
                index_type="IVF_PQ",
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
            )
        except TypeError:
            table.create_index("vector", index_type="IVF_PQ")

    def _measure(
        self,
        table: lancedb.Table,
        nprobes: int = 128,
        refine_factor: int = 10,
    ) -> dict[str, Any]:
        queries = self._generate_vectors(self.num_queries)
        latencies: list[float] = []
        for q in queries:
            start = time.perf_counter()
            table.search(q.tolist()).metric("cosine").limit(10).nprobes(nprobes).refine_factor(refine_factor).to_pandas()
            latencies.append(time.perf_counter() - start)
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(0.95 * len(latencies))]
        disk_mb = self._dir_size_mb(self.uri)
        recall = self._recall_sample(table, queries)
        return {
            "p50_latency_sec": p50,
            "p95_latency_sec": p95,
            "disk_mb": disk_mb,
            "recall_at_10": recall,
            "nprobes": nprobes,
            "refine_factor": refine_factor,
        }

    def _recall_sample(
        self,
        table: lancedb.Table,
        queries: np.ndarray,
        sample_n: int = 5_000,
        k: int = 10,
    ) -> float:
        if len(queries) == 0:
            return 0.0
        total_rows = table.count_rows()
        sample_size = min(sample_n, total_rows)
        sample_indices = self.rng.choice(total_rows, size=sample_size, replace=False)
        sample_set = {int(i) for i in sample_indices}
        try:
            lance_table = table.to_lance()
            sample_arrow = lance_table.take(sample_indices.astype(int).tolist())
            sample_vectors = np.vstack(
                [
                    np.array(v, dtype=np.float32)
                    for v in sample_arrow.column("vector").to_pylist()
                ]
            )
        except Exception:
            # Fallback: read the full table. Only safe for small corpora.
            df = table.to_pandas()
            sample_vectors = np.vstack(
                [np.array(v, dtype=np.float32) for v in df["vector"]]
            )
            sample_indices = np.arange(len(df))
            sample_set = set(sample_indices.tolist())

        scores = queries @ sample_vectors.T
        recall_sum = 0.0
        counted = 0
        for i, q in enumerate(queries):
            ann = (
                table.search(q.tolist())
                .metric("cosine")
                .limit(k)
                .nprobes(128)
                .to_pandas()["id"]
                .tolist()
            )
            ann_in_sample = [aid for aid in ann if aid in sample_set]
            if not ann_in_sample:
                continue
            true_top = set(sample_indices[np.argpartition(scores[i], -k)[-k:]])
            recall_sum += len(set(ann_in_sample) & true_top) / k
            counted += 1
        return recall_sum / max(counted, 1)

    @staticmethod
    def _dir_size_mb(path: Path) -> float:
        total = 0
        for dirpath, _dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = Path(dirpath) / f
                total += fp.stat().st_size
        return total / (1024 * 1024)

    @staticmethod
    def _power_law_project(
        sizes: list[int],
        values: list[float],
        target: int,
    ) -> float:
        """Fit log(value) = a + b*log(size) and predict at ``target``."""
        if len(sizes) < 2:
            return values[-1] * (target / sizes[-1]) if sizes else 0.0
        x = np.log10(np.array(sizes, dtype=np.float64))
        y = np.log10(np.array(values, dtype=np.float64))
        coeffs = np.polyfit(x, y, 1)
        return float(10 ** (coeffs[1] + coeffs[0] * np.log10(target)))

    def run_scale(
        self,
        sizes: list[int] | None = None,
    ) -> dict[str, Any]:
        if sizes is None:
            sizes = [100_000, 1_000_000, 10_000_000]
        results: dict[str, Any] = {"scales": {}}

        for n in sizes:
            table_name = f"scale_{n}"
            scale_uri = self.uri / table_name

            build_start = time.perf_counter()
            table = self._make_table(scale_uri, n, table_name)
            nprobes = max(32, min(512, n // 10_000))
            self._create_index(table, n)
            build_time = time.perf_counter() - build_start

            metrics = self._measure(table, nprobes=nprobes, refine_factor=10)
            metrics["build_time_sec"] = build_time
            results["scales"][str(n)] = metrics

        if results["scales"]:
            sizes_measured = [int(s) for s in results["scales"]]
            p95s = [results["scales"][str(s)]["p95_latency_sec"] for s in sizes_measured]
            disks = [results["scales"][str(s)]["disk_mb"] for s in sizes_measured]
            builds = [results["scales"][str(s)]["build_time_sec"] for s in sizes_measured]
            results["projection_100m"] = {
                "p95_latency_sec": self._power_law_project(sizes_measured, p95s, 100_000_000),
                "disk_mb": self._power_law_project(sizes_measured, disks, 100_000_000),
                "build_time_sec": self._power_law_project(sizes_measured, builds, 100_000_000),
                "method": "log-log_power_law_fit",
                "source_scales": sizes_measured,
                "note": "Extrapolation from measured scales; actual results depend on hardware and tuning.",
            }
        return results

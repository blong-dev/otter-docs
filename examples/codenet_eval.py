"""Reproduce the CodeNet-Python800 clone-detection number.

This is the documented, parameterized procedure behind the
README's CodeNet figure. It is NOT run in CI (needs the ~30 MB
dataset + a live LLM + embedder; CI has none of those). Anyone can
reproduce the number by:

  1. Download the dataset (CDLA-Permissive-2.0, commercial use OK):
       curl -O https://codait-cos-dax.s3.us.cloud-object-storage.\\
       appdomain.cloud/dax-project-codenet/1.0.0/\\
       Project_CodeNet_Python800.tar.gz
       tar xzf Project_CodeNet_Python800.tar.gz

  2. Point this script at the extracted dir + your model endpoints:
       python examples/codenet_eval.py \\
         --root ./Project_CodeNet_Python800 \\
         --llm-url  http://localhost:11434 --llm-model  <model> \\
         --embed-url http://localhost:11435 --embed-model <model> \\
         --out result.json

Resumable: descriptions are cached in --cache-db, so a re-run after
an interruption skips already-described snippets. The frozen
SamplerConfig is written into result.json so the number is
reproducible by construction — same seed + config + models → same
pairs → same score.

The script reports TWO numbers side by side:
  headline   — type-4-enforced positives (structural-divergence gate)
  baseline   — unfiltered same-problem positives
The gap between them is the contamination delta: how much an
unfiltered "same problem = clone" score is just type-1/2 copy-paste.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from otter_docs.clients import OpenAICompatEmbeddingClient, OpenAICompatLLMClient
from otter_docs.describe import SqliteDescriptionCache
from otter_docs.eval import sweep_threshold
from otter_docs.eval_codenet import SamplerConfig, fill_descriptions, sample


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CodeNet-Python800 clone eval")
    ap.add_argument("--root", required=True,
                    help="extracted Project_CodeNet_Python800 dir")
    ap.add_argument("--llm-url", default="http://localhost:11434")
    ap.add_argument("--llm-model", required=True)
    ap.add_argument("--embed-url", default="http://localhost:11435")
    ap.add_argument("--embed-model", required=True)
    ap.add_argument("--embed-dim", type=int, default=768)
    ap.add_argument("--out", default="codenet_result.json")
    ap.add_argument("--cache-db", default="codenet_desc_cache.db",
                    help="persistent description cache (enables resume)")
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--positives", type=int, default=200)
    ap.add_argument("--negatives", type=int, default=200)
    args = ap.parse_args(argv)

    cfg = SamplerConfig(
        seed=args.seed,
        target_positives=args.positives,
        target_negatives=args.negatives,
    )
    rep = sample(args.root, cfg)

    llm = OpenAICompatLLMClient(
        model=args.llm_model, base_url=args.llm_url, default_max_tokens=200
    )
    emb = OpenAICompatEmbeddingClient(
        model=args.embed_model, base_url=args.embed_url, dim=args.embed_dim
    )
    cache = SqliteDescriptionCache(sqlite3.connect(args.cache_db))

    t0 = time.time()
    filtered = fill_descriptions(rep.labeled(), llm, cache)
    unfiltered = fill_descriptions(rep.unfiltered_labeled(), llm, cache)
    describe_secs = round(time.time() - t0, 1)

    sw_f = sweep_threshold(filtered, emb, start=0.50, stop=0.95, step=0.025)
    sw_u = sweep_threshold(unfiltered, emb, start=0.50, stop=0.95, step=0.025)

    out = {
        "config": cfg.__dict__,
        "models": {
            "llm": args.llm_model,
            "embed": args.embed_model,
            "embed_dim": args.embed_dim,
        },
        "counts": {
            "positives_t4": len(rep.positives),
            "positives_unfiltered": len(rep.positives_unfiltered),
            "neg_hard": len(rep.negatives_hard),
            "neg_random": len(rep.negatives_random),
            "problems_used": rep.problems_used,
        },
        "describe_seconds": describe_secs,
        "headline_t4_enforced": sw_f.best.as_dict(),
        "baseline_unfiltered_same_problem": sw_u.best.as_dict(),
        "headline_curve": [m.as_dict() for m in sw_f.all],
    }
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("headline (type-4 enforced):",
          json.dumps(out["headline_t4_enforced"]))
    print("baseline (unfiltered same-problem):",
          json.dumps(out["baseline_unfiltered_same_problem"]))
    delta = (out["baseline_unfiltered_same_problem"]["f1"]
             - out["headline_t4_enforced"]["f1"])
    print(f"contamination delta (baseline - headline F1): {delta:+.3f}")
    print(f"wrote {args.out}  (describe took {describe_secs}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

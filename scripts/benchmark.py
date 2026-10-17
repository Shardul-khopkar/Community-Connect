import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.ai_client import ask_ai

PROMPT = "Cerco ripetizioni di matematica a Genova sabato."


def run_samples(iterations: int, force_local: bool, timeout: float) -> tuple[list[float], list[dict]]:
    times = []
    results = []
    environment = {"AI_BASE_URL": "", "AI_API_KEY": ""} if force_local else {}
    with patch.dict(os.environ, environment):
        for _ in range(iterations):
            started = time.perf_counter()
            result = ask_ai(PROMPT, max_new_tokens=96, timeout=timeout)
            times.append(time.perf_counter() - started)
            results.append(result)
    return times, results


def summarize(label: str, times: list[float], results: list[dict]) -> dict | None:
    amd = [(elapsed, result) for elapsed, result in zip(times, results) if result.get("backend") == "amd"]
    if label == "AMD" and not amd:
        return None
    if label == "AMD":
        times = [elapsed for elapsed, _result in amd]
        results = [result for _elapsed, result in amd]
    total_elapsed = sum(times)
    token_count = sum(int(result.get("tokens", 0) or 0) for result in results)
    server_seconds = sum(float(result.get("latency_s", 0) or 0) for result in results)
    return {
        "backend": label,
        "model": results[-1].get("model", "unknown"),
        "calls": len(results),
        "p50_ms": statistics.median(times) * 1000,
        "requests_s": len(times) / total_elapsed if total_elapsed else 0,
        "tokens_s": token_count / server_seconds if server_seconds > 0 and token_count else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark local fallback and configured AMD inference service.")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    if args.iterations < 1 or args.iterations > 100:
        parser.error("--iterations must be between 1 and 100")

    local_times, local_results = run_samples(args.iterations, force_local=True, timeout=args.timeout)
    rows = [summarize("Local fallback", local_times, local_results)]
    if os.environ.get("AI_BASE_URL") and os.environ.get("AI_API_KEY"):
        cloud_times, cloud_results = run_samples(args.iterations, force_local=False, timeout=args.timeout)
        cloud_row = summarize("AMD", cloud_times, cloud_results)
        rows.append(cloud_row)
        if cloud_row is None:
            print("AMD backend did not return a cloud response; check AI_BASE_URL, tunnel, key, and /health.")
    else:
        print("AMD backend not measured: configure AI_BASE_URL and AI_API_KEY, then rerun.")

    print("\nCommunityConnect benchmark (synthetic prompt; wall-clock measurement)")
    print(f"{'Backend':<18} {'Model':<48} {'Calls':>5} {'P50 ms':>10} {'Req/s':>9} {'Tokens/s':>12}")
    print("-" * 107)
    for row in rows:
        if row is None:
            continue
        token_rate = f"{row['tokens_s']:.2f}" if row["tokens_s"] is not None else "n/a"
        print(f"{row['backend']:<18} {row['model'][:48]:<48} {row['calls']:>5} {row['p50_ms']:>10.2f} {row['requests_s']:>9.2f} {token_rate:>12}")
    print("P50 is client-observed per-call latency; AMD tokens/s uses server-reported generation tokens and latency.")
    print("Local fallback does not generate model tokens, so its token rate is n/a.")


if __name__ == "__main__":
    main()

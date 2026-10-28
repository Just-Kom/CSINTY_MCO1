"""
Run the Sokobot solver on every .txt level in ./maps with a per-level time limit,
in parallel, then print a summary table and save results.csv.

Usage (from the sokobot folder):
    python run_all.py
    python run_all.py --timeout 15 --workers 4
    python run_all.py --maps maps
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def run_once(map_arg, timeout):
    """Run driver on one map arg. Returns (status, seconds, stdout, stderr)."""
    cmd = [sys.executable, "-m", "src.main.driver", map_arg, "raw"]
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.perf_counter() - start, "", ""
    elapsed = time.perf_counter() - start

    out, err = proc.stdout.strip(), proc.stderr.strip()
    if proc.returncode != 0:
        return "ERROR", elapsed, out, err
    if not out:
        return "NO_SOLUTION", elapsed, out, err
    return "SOLVED", elapsed, out, err


def run_map(filename, timeout):
    stem = os.path.splitext(filename)[0]
    # Don't know whether FileReader wants "level1" or "level1.txt", so try the
    # stem first and fall back to the full filename if the map couldn't load.
    status, secs, out, err = run_once(stem, timeout)
    if status == "ERROR" and "could not be loaded" in err:
        status, secs, out, err = run_once(filename, timeout)

    # Moves string is the last line printed by the driver
    moves = out.splitlines()[-1] if out else ""
    return {
        "map": filename,
        "status": status,
        "time_s": round(secs, 2),
        "moves": len(moves.strip("[]'\" ")) if status == "SOLVED" else 0,
        "error": err.splitlines()[-1] if status == "ERROR" and err else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default="maps", help="maps directory")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="parallel processes (fewer = fairer timings)",
    )
    args = ap.parse_args()

    maps_dir = os.path.join(ROOT, args.maps)
    files = sorted(
        (f for f in os.listdir(maps_dir) if f.lower().endswith(".txt")),
        key=natural_key,
    )
    if not files:
        print(f"No .txt maps found in {maps_dir}")
        return

    print(f"Running {len(files)} maps | timeout {args.timeout}s | workers {args.workers}\n")
    results = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_map, f, args.timeout): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            print(f"[{i:>3}/{len(files)}] {r['map']:<25} {r['status']:<12} {r['time_s']:>6.2f}s")

    wall = time.perf_counter() - wall_start
    results.sort(key=lambda r: natural_key(r["map"]))

    # ---- summary ----
    print("\n" + "=" * 62)
    print(f"{'MAP':<25}{'STATUS':<14}{'TIME(s)':>9}{'MOVES':>9}")
    print("-" * 62)
    for r in results:
        print(f"{r['map']:<25}{r['status']:<14}{r['time_s']:>9.2f}{r['moves']:>9}")
    print("=" * 62)

    solved = [r for r in results if r["status"] == "SOLVED"]
    print(f"Solved:      {len(solved)}/{len(results)}")
    for s in ("TIMEOUT", "ERROR", "NO_SOLUTION"):
        n = sum(r["status"] == s for r in results)
        if n:
            print(f"{s.title():<12} {n}")
    if solved:
        print(f"Solve time:  total {sum(r['time_s'] for r in solved):.2f}s, "
              f"avg {sum(r['time_s'] for r in solved) / len(solved):.2f}s")
    print(f"Wall clock:  {wall:.2f}s")

    failed = [r["map"] for r in results if r["status"] != "SOLVED"]
    if failed:
        print("\nNot solved: " + ", ".join(failed))

    csv_path = os.path.join(ROOT, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["map", "status", "time_s", "moves", "error"])
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
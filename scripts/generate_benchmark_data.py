# =================================================================
# IdleFL v2 — Benchmark Data Generator
# Generates IID and non-IID synthetic datasets to benchmark
# FedAvg vs FedProx under controlled data distribution conditions.
#
# v1 paper (IEEE IATMSI-2026) used a single IID linear regression
# dataset. v2 extends this to 4 benchmark configurations:
#
#   Config 1: Sync  + FedAvg  + IID     (replicates v1 baseline)
#   Config 2: Async + FedAvg  + IID     (shows async speedup)
#   Config 3: Sync  + FedAvg  + non-IID (shows accuracy degradation)
#   Config 4: Async + FedProx + non-IID (v2 headline result)
#
# Usage:   python generate_benchmark_data.py
# Output:  ./benchmark_data/ folder with CSV files
# =================================================================

import csv
import os
import random

import numpy as np


def _ensure_out_dir():
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_data")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _iid_linear_10k(out_dir):
    n = 10_000
    x = np.random.normal(0.0, 1.0, size=(n, 2))
    noise = np.random.normal(0.0, 0.5, size=(n,))
    y = 3.0 * x[:, 0] + 2.0 * x[:, 1] + noise
    rows = [[float(x[i, 0]), float(x[i, 1]), float(y[i])] for i in range(n)]
    _write_csv(os.path.join(out_dir, "iid_linear_10k.csv"), ["feature_1", "feature_2", "target"], rows)


def _noniid_linear_devices(out_dir):
    configs = [
        (1, 0.0, 1.0, 0.3),
        (2, 2.0, 1.0, 0.7),
        (3, -2.0, 1.0, 1.2),
        (4, 0.0, 2.0, 0.3),
        (5, 1.0, 0.5, 0.7),
    ]

    n = 2000
    for device_id, mean, std, noise_std in configs:
        x = np.random.normal(mean, std, size=(n, 2))
        noise = np.random.normal(0.0, noise_std, size=(n,))
        y = 3.0 * x[:, 0] + 2.0 * x[:, 1] + noise
        rows = [[float(x[i, 0]), float(x[i, 1]), float(y[i])] for i in range(n)]
        name = f"noniid_linear_device{device_id}.csv"
        _write_csv(os.path.join(out_dir, name), ["feature_1", "feature_2", "target"], rows)


def _sample_logistic_row():
    x = np.random.normal(0.0, 1.0, size=(4,))
    # A simple linear boundary with a bit of stochasticity.
    score = 1.1 * x[0] - 0.9 * x[1] + 0.6 * x[2] - 0.4 * x[3] + np.random.normal(0.0, 0.3)
    p = 1.0 / (1.0 + np.exp(-np.clip(score, -20, 20)))
    y = 1 if random.random() < float(p) else 0
    return x, y


def _noniid_logistic_devices(out_dir):
    targets = [
        (1, 0.50),
        (2, 0.70),
        (3, 0.30),
        (4, 0.80),
        (5, 0.60),
    ]
    n = 2000

    header = ["feature_1", "feature_2", "feature_3", "feature_4", "target"]
    for device_id, desired_pos_rate in targets:
        rows = []
        # Rejection sampling to hit the target class balance.
        max_attempts = n * 200
        attempts = 0
        while len(rows) < n and attempts < max_attempts:
            attempts += 1
            x, y = _sample_logistic_row()
            if y == 1:
                accept_prob = desired_pos_rate / 0.5
            else:
                accept_prob = (1.0 - desired_pos_rate) / 0.5
            accept_prob = max(0.0, min(1.0, float(accept_prob)))
            if random.random() <= accept_prob:
                rows.append([float(x[0]), float(x[1]), float(x[2]), float(x[3]), int(y)])

        if len(rows) < n:
            # Fallback: fill remaining without rejection (keeps script robust).
            while len(rows) < n:
                x, y = _sample_logistic_row()
                rows.append([float(x[0]), float(x[1]), float(x[2]), float(x[3]), int(y)])

        name = f"noniid_logistic_device{device_id}.csv"
        _write_csv(os.path.join(out_dir, name), header, rows)


def main():
    out_dir = _ensure_out_dir()
    _iid_linear_10k(out_dir)
    _noniid_linear_devices(out_dir)
    _noniid_logistic_devices(out_dir)

    print("Generated benchmark_data/:")
    print("  iid_linear_10k.csv — 10000 rows — use for configs 1 and 2")
    print("  noniid_linear_device[1-5].csv — 2000 rows each — configs 3 and 4")
    print("  noniid_logistic_device[1-5].csv — 2000 rows each — logistic benchmark")
    print("")
    print("To reproduce v2 paper results:")
    print("  Config 1 (v1 baseline): upload iid_linear_10k.csv, mu=0")
    print("  Config 2 (async IID):   upload iid_linear_10k.csv, mu=0, threshold=0.7")
    print("  Config 3 (non-IID):     each device sets DATASET_PATH to their noniid file, mu=0")
    print("  Config 4 (FedProx):     each device sets DATASET_PATH to their noniid file, mu=0.01")


if __name__ == "__main__":
    main()


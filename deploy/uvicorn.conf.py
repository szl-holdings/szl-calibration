"""Production Uvicorn tuning for szl-calibration.
Workers: 2*cores+1 capped at 8 - calibration scoring is CPU-bound pure math, so
extra workers only thrash. Override with UVICORN_WORKERS.
"""
import os

import multiprocessing

workers = min(int(os.environ.get("UVICORN_WORKERS", 0)) or (2 * multiprocessing.cpu_count() + 1), 8)
bind = "0.0.0.0:8080"
backlog = 2048
timeout_keep_alive = 30
limit_concurrency = 512
limit_max_requests = 100_000          # recycle workers against slow leaks
limit_max_requests_jitter = 5_000

"""Run many independent experiments across processes (one BLAS thread each so 14 workers
do not fight over cores). Results are identical to a serial run: every run is seeded and
self-contained."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

from .experiment import run_one  # noqa: E402


def _job(kw):
    return run_one(**kw)


def run_many(jobs: list[dict], workers: int | None = None, desc: str = "") -> list[dict]:
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    out = []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(_job, jobs, chunksize=1), 1):
            out.append(r)
            if i % max(1, len(jobs) // 10) == 0 or i == len(jobs):
                print(f"  {desc} {i}/{len(jobs)}", flush=True)
    return out

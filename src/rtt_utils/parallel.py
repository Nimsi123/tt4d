"""Run one function over every clip of the dataset with a pool of worker processes."""

import multiprocessing
import os
import re
from dataclasses import dataclass, field
from multiprocessing import Process, Queue

from rtt_utils.paths import get_home_path


@dataclass
class ParallelConfig:
    """
    Shared configuration for parallel processing scripts that map functions over the dataset.

    This is used by scripts like ball_tracker, calibration, human_pose_tracker, video_thinning, etc.

    Resource modes (mutually exclusive):
    - GPU mode: Provide `resource_ids` as a list of GPU IDs (e.g., [0, 1, 2])
    - CPU mode: Provide `num_resources` as an integer (number of CPU cores to use)

    If both are provided, `resource_ids` takes precedence.
    """

    resource_ids: list[int] = field(default_factory=list)
    """List of resource IDs (GPU IDs) to use for processing. If provided, enables GPU mode."""

    num_resources: int = 0
    """Number of CPU resources to use. If resource_ids is empty and num_resources > 0, enables CPU mode."""

    workers_per_resource: int = 1
    """Number of worker processes per resource (GPU or CPU core)."""

    num_workers: int = 1
    """Total number of worker machines (for distributed processing across machines)."""

    worker_split: int = 0
    """Which worker slice this instance handles (0-indexed, < num_workers)."""

    batch_start: int = 1
    """First batch to process (inclusive)."""

    batch_end: int = 52
    """Last batch to process (inclusive). Batches without a folder are skipped."""

    test: bool = False
    """Run on `pipeline_outputs_test/` instead of `pipeline_outputs/shared/`, over every batch found there."""

    def get_root_dir(self) -> str:
        """Returns the directory that holds the `batch_(un)clipped_long_{i}` folders."""
        if self.test:
            return f"{get_home_path()}/pipeline_outputs_test"
        return f"{get_home_path()}/pipeline_outputs/shared"

    def get_batch_indices(self) -> list[int]:
        """
        Returns the batch numbers a script should walk.
        - Test mode: every `batch_unclipped_long_{i}` / `batch_clipped_long_{i}` found under the test root.
        - Otherwise: `batch_start` through `batch_end`, both included.
        """
        if not self.test:
            return list(range(self.batch_start, self.batch_end + 1))
        indices = set()
        for name in os.listdir(self.get_root_dir()):
            match = re.fullmatch(r"batch_(?:un)?clipped_long_(\d+)", name)
            if match:
                indices.add(int(match.group(1)))
        return sorted(indices)

    def get_resource_ids(self) -> list[int]:
        """
        Returns the list of resource IDs to use.
        - If resource_ids is provided (non-empty), returns that list (GPU mode).
        - Otherwise, generates a list [0, 1, ..., num_resources-1] (CPU mode).
        """
        if self.resource_ids:
            return self.resource_ids
        elif self.num_resources > 0:
            return list(range(self.num_resources))
        else:
            return [0]  # default to single resource

    def total_workers(self) -> int:
        """Returns the total number of worker processes."""
        return len(self.get_resource_ids()) * self.workers_per_resource

    def is_gpu_mode(self) -> bool:
        """Returns True if running in GPU mode (resource_ids explicitly provided)."""
        return len(self.resource_ids) > 0

    def slice_tasks(self, tasks):
        """Returns this machine's share of `tasks` (slice `worker_split` of `num_workers` equal slices)."""
        task_slice = tasks[
            int(len(tasks) * self.worker_split / self.num_workers) : int(
                len(tasks) * (self.worker_split + 1) / self.num_workers
            )
        ]
        print("len(task_slice): ", len(task_slice))
        return task_slice


def run_parallel(tasks, task_fn, config: ParallelConfig, setup=None):
    """
    Process `tasks` with `len(resource_ids) * workers_per_resource` worker processes fed from one queue.

    Each worker calls `task_fn(task)`, or `task_fn(task, state)` where `state = setup(resource_id)` ran once in
    that worker before its first task (use it to load a model). In GPU mode a worker sees only its GPU through
    CUDA_VISIBLE_DEVICES, so code inside it addresses the GPU as "cuda:0". A task that raises is reported and
    skipped. Call `multiprocessing.set_start_method("spawn")` first when the workers use CUDA.
    """
    resource_ids = config.get_resource_ids()
    total_workers = len(resource_ids) * config.workers_per_resource

    print(f"Total tasks            : {len(tasks)}")
    print(f"Resources              : {list(resource_ids)}")
    print(f"Workers per resource   : {config.workers_per_resource}")
    print(f"Total worker processes : {total_workers}")

    task_queue = Queue()
    for t in tasks:
        task_queue.put(t)
    for _ in range(total_workers):
        task_queue.put(None)  # one sentinel per worker

    processes = []
    for resource_id in resource_ids:
        for _ in range(config.workers_per_resource):
            p = Process(target=_worker, args=(resource_id, task_queue, task_fn, setup, config.is_gpu_mode()))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()

    print("All tasks completed.")


def _worker(resource_id, task_queue, task_fn, setup, gpu_mode):
    if gpu_mode:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(resource_id)
    state = setup(resource_id) if setup is not None else None
    while True:
        task = task_queue.get()
        if task is None:
            break
        try:
            if setup is None:
                task_fn(task)
            else:
                task_fn(task, state)
            print(f"[Worker {resource_id}] Completed {task}", flush=True)
        except Exception as e:
            print(f"[Worker {resource_id}] Failed on {task}: {e}", flush=True)
    print(f"[Worker {resource_id}] No more work. Shutting down.", flush=True)


def recommended_workers(buffer=0.15, max_workers=None) -> tuple[float, int]:
    """Return (1-minute load per CPU, worker count that leaves `buffer` of the machine idle)."""
    # number of logical CPUs
    ncpu = multiprocessing.cpu_count()
    if max_workers is None:
        max_workers = ncpu

    # get 1-minute load average
    load1, _, _ = os.getloadavg()
    load_per_cpu = load1 / ncpu

    # heuristic: leave some headroom based on current load
    # available fraction = (1 - load_per_cpu), then apply buffer
    available_frac = max(0.0, 1.0 - load_per_cpu - buffer)
    workers = int(max_workers * available_frac)

    # at least 1 worker if machine not totally saturated
    workers = max(1, min(workers, max_workers))
    return load_per_cpu, workers

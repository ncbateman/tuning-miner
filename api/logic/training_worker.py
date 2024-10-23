import queue
import threading
from uuid import UUID

import docker
from fiber.logging_utils import get_logger

from core.models.utility_models import Job
from core.models.utility_models import JobStatus
from miner.logic.job_handler import start_tuning_container


logger = get_logger(__name__)


# one of the only classes I promise
class TrainingWorker:
    def __init__(self):
        logger.info("STARTING A TRAINING WORKER")
        self.job_queue: queue.Queue[Job] = queue.Queue()
        self.job_store: dict[str, Job] = {}
        # Why do we need a separate thread here? dangerous - when I removed it it didn't work any more
        # will dig into why
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.docker_client = docker.from_env()

    def _worker(self):
        while True:
            job = self.job_queue.get()
            if job is None:
                break
            try:
                start_tuning_container(job)
                job.status = JobStatus.COMPLETED
            except Exception as e:
                logger.error(f"Error processing job {job.job_id}: {str(e)}")
                job.status = JobStatus.FAILED
                job.error_message = str(e)
            finally:
                self.job_queue.task_done()

    def enqueue_job(self, job: Job):
        self.job_queue.put(job)
        self.job_store[job.job_id] = job

    def get_status(self, job_id: UUID) -> JobStatus:
        job = self.job_store.get(str(job_id))
        # this doesn't match the typehint
        return job.status if job else "Not Found"

    def shutdown(self):
#        self.job_queue.put(None)
        self.thread.join()
        self.docker_client.close()

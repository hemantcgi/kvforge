# tests/test_studio_job_manager.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from studio.job_manager import JobManager, JobStatus, DuplicateJobError


def test_create_job_returns_job_id():
    jm = JobManager()
    job_id = jm.create("usecase3_squad", "train")
    assert job_id is not None and len(job_id) > 0


def test_get_job_returns_created_job():
    jm = JobManager()
    job_id = jm.create("usecase3_squad", "train")
    job = jm.get(job_id)
    assert job["uc_id"] == "usecase3_squad"
    assert job["step"] == "train"
    assert job["status"] == JobStatus.RUNNING


def test_duplicate_uc_raises():
    jm = JobManager()
    jm.create("usecase3_squad", "train")
    with pytest.raises(DuplicateJobError):
        jm.create("usecase3_squad", "prs-eval")


def test_different_ucs_allowed_concurrently():
    jm = JobManager()
    jm.create("usecase3_squad", "train")
    job_id2 = jm.create("usecase1_customer_support", "train")
    assert job_id2 is not None


def test_complete_job_releases_lock():
    jm = JobManager()
    job_id = jm.create("usecase3_squad", "train")
    jm.complete(job_id, exit_code=0)
    assert jm.get(job_id)["status"] == JobStatus.DONE
    # Now same UC can start a new job
    job_id2 = jm.create("usecase3_squad", "prs-eval")
    assert job_id2 is not None


def test_fail_job():
    jm = JobManager()
    job_id = jm.create("usecase3_squad", "train")
    jm.fail(job_id, "CUDA out of memory")
    assert jm.get(job_id)["status"] == JobStatus.FAILED
    assert "CUDA" in jm.get(job_id)["error"]


def test_append_log_line():
    jm = JobManager()
    job_id = jm.create("usecase3_squad", "train")
    jm.append_log(job_id, "epoch 1/3 complete")
    jm.append_log(job_id, "epoch 2/3 complete")
    assert len(jm.get(job_id)["last_lines"]) == 2


def test_log_capped_at_100_lines():
    jm = JobManager()
    job_id = jm.create("usecase3_squad", "train")
    for i in range(150):
        jm.append_log(job_id, f"line {i}")
    assert len(jm.get(job_id)["last_lines"]) == 100


def test_list_active_returns_running_jobs():
    jm = JobManager()
    jm.create("usecase3_squad", "train")
    active = jm.list_active()
    assert len(active) == 1
    assert active[0]["uc_id"] == "usecase3_squad"


def test_stop_job_marks_stopped():
    jm = JobManager()
    job_id = jm.create("usecase3_squad", "train")
    jm.stop(job_id)
    assert jm.get(job_id)["status"] == JobStatus.STOPPED

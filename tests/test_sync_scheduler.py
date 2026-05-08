import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_scheduler_runs_job():
    from core.sync_scheduler import APSchedulerBackend
    results = []
    sched = APSchedulerBackend()
    sched.start()
    job_id = sched.schedule("uc1", interval_minutes=1, fn=lambda: results.append(1))
    sched.trigger_now(job_id)
    time.sleep(0.5)
    sched.stop()
    assert len(results) >= 1


def test_scheduler_cancel_job():
    from core.sync_scheduler import APSchedulerBackend
    results = []
    sched = APSchedulerBackend()
    sched.start()
    job_id = sched.schedule("uc1", interval_minutes=60, fn=lambda: results.append(1))
    sched.cancel(job_id)
    jobs = sched.list_jobs()
    assert not any(j.job_id == job_id for j in jobs)
    sched.stop()


def test_scheduler_list_jobs():
    from core.sync_scheduler import APSchedulerBackend
    sched = APSchedulerBackend()
    sched.start()
    job_id = sched.schedule("uc2", interval_minutes=30, fn=lambda: None)
    jobs = sched.list_jobs()
    assert any(j.uc_name == "uc2" and j.interval_minutes == 30 for j in jobs)
    sched.stop()


def test_sync_scheduler_protocol_satisfied():
    from core.sync_scheduler import APSchedulerBackend, SyncScheduler
    sched = APSchedulerBackend()
    assert isinstance(sched, SyncScheduler)

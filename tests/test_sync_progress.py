# tests/test_sync_progress.py
import asyncio, pytest
from sync.progress import ProgressBus

@pytest.mark.asyncio
async def test_publish_and_receive():
    bus = ProgressBus()
    await bus.publish("conn1", {"event": "progress", "files_done": 5})
    event = await asyncio.wait_for(bus.get("conn1"), timeout=1.0)
    assert event["files_done"] == 5

@pytest.mark.asyncio
async def test_complete_event_clears_queue():
    bus = ProgressBus()
    await bus.publish("conn2", {"event": "complete", "files_done": 10})
    event = await asyncio.wait_for(bus.get("conn2"), timeout=1.0)
    assert event["event"] == "complete"

@pytest.mark.asyncio
async def test_separate_connectors_isolated():
    bus = ProgressBus()
    await bus.publish("conn-a", {"event": "progress", "msg": "a"})
    await bus.publish("conn-b", {"event": "progress", "msg": "b"})
    a = await asyncio.wait_for(bus.get("conn-a"), timeout=1.0)
    b = await asyncio.wait_for(bus.get("conn-b"), timeout=1.0)
    assert a["msg"] == "a"
    assert b["msg"] == "b"

import threading
import time


class FakeWorker:
    def __init__(self) -> None:
        self.completed = False

    def start(self) -> None:
        thread = threading.Thread(target=self._run)
        thread.start()

    def _run(self) -> None:
        delay = 0.08 if time.monotonic_ns() % 97 < 10 else 0.04
        time.sleep(delay)
        self.completed = True


def test_worker_completes() -> None:
    worker = FakeWorker()

    worker.start()
    time.sleep(0.05)

    assert worker.completed

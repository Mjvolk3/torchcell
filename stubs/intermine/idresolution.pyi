from _typeshed import Incomplete

def get_json(service, path, key): ...

ONE_MINUTE: int
COMPLETED: Incomplete

class Job:
    INITIAL_DECAY: float
    INITIAL_BACKOFF: float
    MAX_BACKOFF = ONE_MINUTE
    service: Incomplete
    uid: Incomplete
    status: Incomplete
    backoff: Incomplete
    decay: Incomplete
    max_backoff: Incomplete
    def __init__(self, service, uid) -> None: ...
    def poll(self): ...
    def fetch_status(self): ...
    def delete(self) -> None: ...
    def fetch_results(self): ...

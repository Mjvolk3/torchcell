from _typeshed import Incomplete

def openAnything(source): ...

class ReadableException(Exception):
    message: Incomplete
    cause: Incomplete
    def __init__(self, message, cause: Incomplete | None = ...) -> None: ...

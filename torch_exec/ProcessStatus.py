import enum


class ProcessStatus(enum.IntEnum):
    """Enum for process status codes"""
    FINISH = 233
    RETRY = 123
    KILLED = -9
    KILLED_ALT = 255
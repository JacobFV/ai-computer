from functools import wraps
from typing import Any

def micro_op(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)
    setattr(MicroOps, func.__name__, wrapper)
    return wrapper

class MicroOps:
    def __init__(self, memory: "Memory"):
        self.memory = memory

    @micro_op
    def mov(self, destination_address: str, source_address: str) -> None:
        data = self.memory.read(source_address)
        self.memory.write(destination_address, data)

    @micro_op
    def write(self, address: str, text: str) -> None:
        self.memory.write(address, text)

    @micro_op
    def clear(self, address: str) -> None:
        self.memory.write(address, "")

    @micro_op
    def insert(self, address: str, text: str) -> None:
        self.memory.insert(address, text)

    @micro_op
    def delete(self, address: str, length: int = 1) -> None:
        self.memory.delete(address, length)

    @micro_op
    def jump(self, core: "Core", address: str) -> None:
        core.instruction_pointer = int(address)

    @micro_op
    def no_op(self) -> None:
        pass

    # Add additional micro-ops as needed

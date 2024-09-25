import asyncio
from abc import ABC, abstractmethod

from ai_computer.memory import Memory


class Peripheral(ABC):
    def __init__(self, memory: "Memory"):
        self.memory = memory
        asyncio.create_task(self.run())

    @abstractmethod
    async def run(self):
        pass

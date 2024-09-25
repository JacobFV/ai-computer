from ai_computer.memory import Memory
from ai_computer.peripherals.base import Peripheral
import asyncio


class TickCounter(Peripheral):
    def __init__(self, memory: "Memory", address: str):
        self.address = address  # Address in memory where tick count is stored
        self.tick_count = 0
        self.running = True
        super().__init__(memory)

    async def run(self):
        while self.running:
            self.tick_count += 1
            self.memory.write(self.address, str(self.tick_count))
            await asyncio.sleep(1)  # Increment every second

    def stop(self):
        self.running = False

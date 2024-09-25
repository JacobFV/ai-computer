from ai_computer.memory import Memory
from ai_computer.peripherals.base import Peripheral
import asyncio
from datetime import datetime


class Clock(Peripheral):
    def __init__(self, memory: "Memory", address: str):
        self.address = address  # Address in memory where time is stored
        self.running = True
        super().__init__(memory)

    async def run(self):
        while self.running:
            current_time = datetime.now().isoformat()
            self.memory.write(self.address, current_time)
            await asyncio.sleep(1)  # Update every second

    def stop(self):
        self.running = False

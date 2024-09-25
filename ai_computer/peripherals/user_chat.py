from ai_computer.memory import Memory
from ai_computer.peripherals.base import Peripheral
import asyncio


class UserChat(Peripheral):
    def __init__(self, memory: "Memory", input_address: str, output_address: str):
        self.input_address = input_address
        self.output_address = output_address
        super().__init__(memory)

    async def run(self):
        while True:
            user_input = await self.get_user_input()
            self.memory.write(self.input_address, user_input)
            # Wait for the output to be written by the processor
            while self.memory.read(self.output_address) == "":
                await asyncio.sleep(0.1)
            response = self.memory.read(self.output_address)
            print(f"Processor: {response}")
            # Clear the output for the next message
            self.memory.write(self.output_address, "")

    async def get_user_input(self) -> str:
        loop = asyncio.get_event_loop()
        user_input = await loop.run_in_executor(None, input, "You: ")
        return user_input

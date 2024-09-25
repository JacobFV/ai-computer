from ai_computer.memory import Memory
from ai_computer.peripherals.base import Peripheral
import asyncio
import ell


class GPT4Chat(Peripheral):
    def __init__(self, memory: "Memory", input_address: str, output_address: str):
        self.input_address = input_address
        self.output_address = output_address
        super().__init__(memory)

    async def run(self):
        while True:
            prompt = self.memory.read(self.input_address)
            if prompt != "":
                response = await self.generate_response(prompt)
                self.memory.write(self.output_address, response)
                # Clear the input for the next prompt
                self.memory.write(self.input_address, "")
            await asyncio.sleep(0.5)

    async def generate_response(self, prompt: str) -> str:
        # Use ell to interact with GPT-4
        @ell.simple(model="gpt-4")
        def chat():
            """You are GPT-4."""
            return prompt

        response = chat()
        return response

from ai_computer.memory import Memory
from ai_computer.peripherals.base import Peripheral
import asyncio
import subprocess


class BashShell(Peripheral):
    def __init__(self, memory: "Memory", input_address: str, output_address: str):
        self.input_address = input_address
        self.output_address = output_address
        super().__init__(memory)

    async def run(self):
        while True:
            command = self.memory.read(self.input_address)
            if command != "":
                output = await self.execute_command(command)
                self.memory.write(self.output_address, output)
                # Clear the input for the next command
                self.memory.write(self.input_address, "")
            await asyncio.sleep(0.5)

    async def execute_command(self, command: str) -> str:
        loop = asyncio.get_event_loop()

        def run_cmd():
            try:
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True
                )
                return result.stdout + result.stderr
            except Exception as e:
                return str(e)

        output = await loop.run_in_executor(None, run_cmd)
        return output

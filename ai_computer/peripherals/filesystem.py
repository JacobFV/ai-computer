from ai_computer.memory import Memory
from ai_computer.peripherals.base import Peripheral
import asyncio
import os


class FileSystemDriver(Peripheral):
    def __init__(self, memory: "Memory", operations_address: str):
        self.operations_address = operations_address  # Address for operations
        super().__init__(memory)

    async def run(self):
        while True:
            operation_line = self.memory.read(self.operations_address)
            if operation_line != "":
                # Parse the operation (expecting JSON formatted string)
                try:
                    operation = eval(
                        operation_line
                    )  # Replace with json.loads if input is JSON
                    cmd = operation.get("cmd")
                    path = operation.get("path")
                    result_address = operation.get("result_address")
                    if cmd == "read":
                        content = await self.read_file(path)
                        self.memory.write(str(result_address), content)
                    elif cmd == "write":
                        content = operation.get("content")
                        await self.write_file(path, content)
                    # Handle other file operations as needed
                except Exception as e:
                    print(f"FileSystemDriver error: {e}")
                # Clear the operation line
                self.memory.write(self.operations_address, "")
            await asyncio.sleep(0.5)

    async def read_file(self, path: str) -> str:
        loop = asyncio.get_event_loop()

        def read():
            if os.path.exists(path):
                with open(path, "r") as f:
                    return f.read()
            else:
                return "File not found"

        content = await loop.run_in_executor(None, read)
        return content

    async def write_file(self, path: str, content: str) -> None:
        loop = asyncio.get_event_loop()

        def write():
            with open(path, "w") as f:
                f.write(content)

        await loop.run_in_executor(None, write)

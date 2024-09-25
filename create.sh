#!/bin/bash

# Script to generate the AI Computer codebase

# Exit immediately if a command exits with a non-zero status
set -e

# Define the project root
PROJECT_ROOT="ai_computer"

# Create the main project directory
mkdir -p "$PROJECT_ROOT"

# Navigate into the project directory
cd "$PROJECT_ROOT"

# Create peripherals directory
mkdir -p peripherals

# Function to create files with content
create_file() {
    local filepath=$1
    local content=$2
    echo "$content" > "$filepath"
}

# Create memory.py
create_file "memory.py" "$(cat << 'EOL'
from typing import List, Tuple, Optional, Dict
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

class Memory:
    def __init__(self):
        self.lines: List[str] = [""] * 1000000  # Initialize memory with 1 million lines
        # Initialize system memory layout
        self.lines[0] = "0"  # Instruction pointer
        self.lines[1] = ""   # Return value
        self.lines[2] = "0"  # Context pointer
        self.lines[3] = "[]" # Interrupt table

        # Load models for semantic matching
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.embeddings_cache: Dict[int, np.ndarray] = {}  # Cache embeddings for performance

    def read(self, address: str) -> str:
        line, col = self.parse_address(address)
        if col is not None:
            if line < len(self.lines):
                line_content = self.lines[line]
                if col < len(line_content):
                    return line_content[col]
                else:
                    return ""
            else:
                return ""
        else:
            if line < len(self.lines):
                return self.lines[line]
            else:
                return ""

    def write(self, address: str, text: str) -> None:
        line, col = self.parse_address(address)
        if col is not None:
            if line >= len(self.lines):
                self.lines.extend([""] * (line - len(self.lines) + 1))
            line_content = self.lines[line]
            if len(line_content) <= col:
                line_content = line_content.ljust(col)
            line_content = line_content[:col] + text + line_content[col + len(text):]
            self.lines[line] = line_content
        else:
            if line >= len(self.lines):
                self.lines.extend([""] * (line - len(self.lines) + 1))
            self.lines[line] = text

    def insert(self, address: str, text: str) -> None:
        line, col = self.parse_address(address)
        if col is not None:
            if line < len(self.lines):
                line_content = self.lines[line]
                line_content = line_content[:col] + text + line_content[col:]
                self.lines[line] = line_content
            else:
                self.lines.extend([""] * (line - len(self.lines)))
                self.lines[line] = text
        else:
            if line <= len(self.lines):
                self.lines.insert(line, text)
            else:
                self.lines.extend([""] * (line - len(self.lines)))
                self.lines.append(text)

    def delete(self, address: str, length: int = 1) -> None:
        line, col = self.parse_address(address)
        if col is not None:
            if line < len(self.lines):
                line_content = self.lines[line]
                line_content = line_content[:col] + line_content[col + length:]
                self.lines[line] = line_content
        else:
            if line < len(self.lines):
                del self.lines[line]

    def parse_address(self, address: str) -> Tuple[int, Optional[int]]:
        if ":" in address:
            line_str, col_str = address.split(":")
            line = int(line_str)
            col = int(col_str)
            return line, col
        else:
            line = int(address)
            return line, None

    def append(self, text: str) -> int:
        self.lines.append(text)
        return len(self.lines) - 1  # Return the address of the appended line

    def resolve_pointer(self, pointer_str: str) -> str:
        # Implement pointer functions
        if pointer_str.startswith("exact_match("):
            match = re.match(r'exact_match\("(.+?)"(?:,\s*occurrence=(\d+))?\)', pointer_str)
            if match:
                query = match.group(1)
                occurrence = int(match.group(2)) if match.group(2) else 0
                return self.exact_match(query, occurrence)
        elif pointer_str.startswith("regex_match("):
            match = re.match(r'regex_match\("(.+?)"(?:,\s*occurrence=(\d+))?\)', pointer_str)
            if match:
                regex = match.group(1)
                occurrence = int(match.group(2)) if match.group(2) else 0
                return self.regex_match(regex, occurrence)
        elif pointer_str.startswith("tfidf_match("):
            match = re.match(r'tfidf_match\("(.+?)"(?:,\s*k=(\d+))?\)', pointer_str)
            if match:
                query = match.group(1)
                k = int(match.group(2)) if match.group(2) else 1
                return self.tfidf_match(query, k)
        elif pointer_str.startswith("semantic_match("):
            match = re.match(r'semantic_match\("(.+?)"(?:,\s*k=(\d+))?\)', pointer_str)
            if match:
                query = match.group(1)
                k = int(match.group(2)) if match.group(2) else 1
                return self.semantic_match(query, k)
        elif pointer_str.startswith("meta_pointer("):
            match = re.match(r'meta_pointer\((.+?)\)', pointer_str)
            if match:
                inner_pointer = match.group(1)
                resolved_pointer = self.resolve_pointer(inner_pointer)
                return resolved_pointer
        else:
            # Invalid pointer
            return "0"

    def exact_match(self, query: str, occurrence: int = 0) -> str:
        matches = [i for i, line in enumerate(self.lines) if line == query]
        if len(matches) > occurrence:
            return str(matches[occurrence])
        else:
            return "0"

    def regex_match(self, regex: str, occurrence: int = 0) -> str:
        pattern = re.compile(regex)
        matches = [i for i, line in enumerate(self.lines) if pattern.search(line)]
        if len(matches) > occurrence:
            return str(matches[occurrence])
        else:
            return "0"

    def tfidf_match(self, query: str, k: int = 1) -> str:
        vectorizer = TfidfVectorizer()
        documents = self.lines
        tfidf_matrix = vectorizer.fit_transform(documents)
        query_vec = vectorizer.transform([query])
        cosine_similarities = np.dot(tfidf_matrix, query_vec.T).toarray().ravel()
        top_indices = cosine_similarities.argsort()[-k:][::-1]
        if len(top_indices) > 0:
            return str(top_indices[0])
        else:
            return "0"

    def semantic_match(self, query: str, k: int = 1) -> str:
        query_embedding = self.get_embedding(query)
        similarities = []
        for idx, line in enumerate(self.lines):
            line_embedding = self.get_embedding(line)
            if np.linalg.norm(query_embedding) == 0 or np.linalg.norm(line_embedding) == 0:
                similarity = 0
            else:
                similarity = np.dot(query_embedding, line_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(line_embedding))
            similarities.append((idx, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        if len(similarities) > 0:
            return str(similarities[0][0])
        else:
            return "0"

    def get_embedding(self, text: str) -> np.ndarray:
        # Check cache
        text_hash = hash(text)
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
        self.embeddings_cache[text_hash] = embedding
        return embedding
EOL
)"

# Create micro_ops.py
create_file "micro_ops.py" "$(cat << 'EOL'
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
EOL
)"

# Create interrupts.py
create_file "interrupts.py" "$(cat << 'EOL'
from typing import List, Dict, Any, Callable

class Interrupts:
    def __init__(self, processor: "Processor"):
        self.processor = processor
        self.memory = processor.memory
        self.interrupt_table: List[Dict[str, Any]] = []

    def add_interrupt(self, name: str, condition: Callable[[], bool], goto_address: int, priority: int = 0) -> None:
        interrupt = {
            "name": name,
            "condition": condition,
            "goto_address": goto_address,
            "priority": priority
        }
        self.interrupt_table.append(interrupt)
        # Sort interrupts by priority (higher number means higher priority)
        self.interrupt_table.sort(key=lambda x: x["priority"], reverse=True)

    def remove_interrupt(self, name: str) -> None:
        self.interrupt_table = [i for i in self.interrupt_table if i["name"] != name]

    def trigger_interrupt(self, name: str) -> None:
        for interrupt in self.interrupt_table:
            if interrupt["name"] == name:
                # Directly set the instruction pointer of all cores
                for core in self.processor.cores:
                    core.instruction_pointer = interrupt["goto_address"]

    def check_interrupts(self, core: "Core") -> bool:
        for interrupt in self.interrupt_table:
            if interrupt["condition"]():
                # Handle interrupt
                core.instruction_pointer = interrupt["goto_address"]
                return True  # Interrupt handled
        return False  # No interrupt
EOL
)"

# Create processor.py
create_file "processor.py" "$(cat << 'EOL'
import asyncio
from typing import List, Dict, Any
import ell
from micro_ops import MicroOps
from interrupts import Interrupts

class Core:
    def __init__(self, core_id: int, memory: "Memory", processor: "Processor"):
        self.core_id = core_id
        self.memory = memory
        self.processor = processor
        self.instruction_pointer = int(self.memory.read("0"))  # Each core has its own IP
        self.no_op_count = 0  # For tracking no_op calls

    def fetch_instruction(self) -> str:
        address = self.instruction_pointer
        instruction = self.memory.read(str(address))
        return instruction

    async def execute(self):
        while True:
            # Check for interrupts
            interrupt_handled = self.processor.check_interrupts(self)
            if interrupt_handled:
                continue

            # Fetch and execute instruction
            instruction = self.fetch_instruction()
            if instruction.strip() == "":
                # No operation, increment IP
                self.instruction_pointer += 1
                continue

            # Parse and execute instruction
            await self.processor.execute_instruction(self, instruction)

    def pause(self):
        # Implementation for pausing the core
        pass

    def resume(self):
        # Implementation for resuming the core
        pass

class Processor:
    def __init__(self, processor_id: int, memory: "Memory", num_cores: int = 4):
        self.processor_id = processor_id
        self.memory = memory
        self.cores: List[Core] = [Core(i, memory, self) for i in range(num_cores)]
        self.context_pointer = int(self.memory.read("2"))  # Shared among cores
        self.interrupts = Interrupts(self)
        self.micro_ops = MicroOps(memory)
        # Load the language model for instruction parsing
        self.model = ell.simple(model="gpt-4")

    def add_interrupt(self, interrupt: Dict[str, Any]) -> None:
        self.interrupts.add_interrupt(**interrupt)

    def remove_interrupt(self, name: str) -> None:
        self.interrupts.remove_interrupt(name)

    def check_interrupts(self, core: Core) -> bool:
        return self.interrupts.check_interrupts(core)

    async def execute_instruction(self, core: Core, instruction: str):
        # Use the context for the core
        context_address = str(self.context_pointer)
        context = self.memory.read(context_address)

        # Prepare the prompt for the language model
        prompt = f"""
Instruction: {instruction}
Context: {context}

Parse the instruction and output the micro-operation and parameters in JSON format.
Example format:
{{
    "micro_op": "write",
    "params": {{
        "address": "100",
        "text": "Hello, World!"
    }}
}}
"""

        # Use ell to get the structured output
        @ell.llm(model="gpt-4", parse="json")
        def parse_instruction(prompt: str) -> Any:
            return prompt

        result = parse_instruction(prompt)

        if result is None or "micro_op" not in result:
            print(f"Error parsing instruction at line {core.instruction_pointer}: {instruction}")
            core.instruction_pointer += 1
            return

        micro_op_name = result["micro_op"]
        params = result["params"]

        # Execute the micro-op
        micro_op = getattr(self.micro_ops, micro_op_name, None)
        if micro_op:
            # Handle parameters that might be pointers
            resolved_params = self.resolve_params(params)
            if "core" in micro_op.__code__.co_varnames:
                micro_op(core, **resolved_params)
            else:
                micro_op(**resolved_params)
        else:
            print(f"Unknown micro-operation: {micro_op_name}")

        # Move to next instruction
        core.instruction_pointer += 1

    def resolve_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Resolve addresses and pointers
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("pointer:"):
                # Resolve pointer
                pointer_str = value[len("pointer:"):]
                address = self.memory.resolve_pointer(pointer_str)
                resolved[key] = address
            else:
                resolved[key] = value
        return resolved

    async def run(self):
        tasks = [core.execute() for core in self.cores]
        await asyncio.gather(*tasks)
EOL
)"

# Create peripherals/base.py
create_file "peripherals/base.py" "$(cat << 'EOL'
import asyncio
from abc import ABC, abstractmethod

class Peripheral(ABC):
    def __init__(self, memory: "Memory"):
        self.memory = memory
        asyncio.create_task(self.run())

    @abstractmethod
    async def run(self):
        pass
EOL
)"

# Create peripherals/tick_counter.py
create_file "peripherals/tick_counter.py" "$(cat << 'EOL'
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
EOL
)"

# Create peripherals/clock.py
create_file "peripherals/clock.py" "$(cat << 'EOL'
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
EOL
)"

# Create peripherals/user_chat.py
create_file "peripherals/user_chat.py" "$(cat << 'EOL'
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
EOL
)"

# Create peripherals/gpt4_chat.py
create_file "peripherals/gpt4_chat.py" "$(cat << 'EOL'
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
EOL
)"

# Create peripherals/bash_shell.py
create_file "peripherals/bash_shell.py" "$(cat << 'EOL'
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
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                return result.stdout + result.stderr
            except Exception as e:
                return str(e)
        output = await loop.run_in_executor(None, run_cmd)
        return output
EOL
)"

# Create peripherals/filesystem.py
create_file "peripherals/filesystem.py" "$(cat << 'EOL'
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
                    operation = eval(operation_line)  # Replace with json.loads if input is JSON
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
EOL
)"

# Create peripherals/__init__.py
create_file "peripherals/__init__.py" "$(cat << 'EOL'
# Init file for peripherals package
EOL
)"

# Create main.py
create_file "main.py" "$(cat << 'EOL'
import asyncio
from ai_computer.memory import Memory
from ai_computer.processor import Processor
from ai_computer.peripherals.tick_counter import TickCounter
from ai_computer.peripherals.clock import Clock
from ai_computer.peripherals.user_chat import UserChat
from ai_computer.peripherals.gpt4_chat import GPT4Chat
from ai_computer.peripherals.bash_shell import BashShell
from ai_computer.peripherals.filesystem import FileSystemDriver

async def main():
    memory = Memory()
    processor = Processor(0, memory)
    micro_ops = processor.micro_ops  # MicroOps instance
    interrupts = processor.interrupts

    # Initialize peripherals
    peripherals = [
        TickCounter(memory, address="10000"),
        Clock(memory, address="10001"),
        UserChat(memory, input_address="10002", output_address="10003"),
        GPT4Chat(memory, input_address="10004", output_address="10005"),
        BashShell(memory, input_address="10006", output_address="10007"),
        FileSystemDriver(memory, operations_address="10010")
    ]

    # Add syscall interrupt
    SYS_CALL_HANDLER_ADDRESS = 5000  # Example address for syscall handler

    def syscall_condition():
        # Check if the current instruction is 'trigger_interrupt("syscall")'
        for core in processor.cores:
            instruction = memory.read(str(core.instruction_pointer))
            if 'trigger_interrupt("syscall")' in instruction:
                return True
        return False

    interrupts.add_interrupt(
        name="syscall",
        condition=syscall_condition,
        goto_address=SYS_CALL_HANDLER_ADDRESS,
        priority=10
    )

    # Add the syscall handler at the specified address
    memory.write(str(SYS_CALL_HANDLER_ADDRESS), "handle_syscall()")

    # Implement the syscall handler in the MicroOps class
    def handle_syscall():
        # Example implementation of syscall handling
        print("Syscall handler executed.")
        # For simplicity, we'll just increment the instruction pointer
        for core in processor.cores:
            core.instruction_pointer += 1

    # Add handle_syscall to MicroOps
    setattr(processor.micro_ops, "handle_syscall", handle_syscall)

    # Load a simple program into memory starting at a specific address
    program_start = 100001  # User process start address
    memory.write(str(program_start), 'write("10004", "Hello GPT-4!")')  # Write to GPT-4 input
    memory.write(str(program_start + 1), 'trigger_interrupt("syscall")')  # Trigger syscall
    memory.write(str(program_start + 2), f'jump("{program_start}")')  # Loop back

    # Set the processor's context pointer to the program start
    processor.context_pointer = program_start
    # Set the cores' instruction pointers to the program start
    for core in processor.cores:
        core.instruction_pointer = program_start

    # Run the processor
    await processor.run()

# Initialize ell
import ell
ell.init(verbose=True)

# Start the asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())
EOL
)"

# Create requirements.txt
create_file "requirements.txt" "$(cat << 'EOL'
pydantic
numpy
transformers
ell
openai
asyncio
scikit-learn
torch
EOL
)"

echo "AI Computer codebase generated successfully!"
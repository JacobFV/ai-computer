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
        FileSystemDriver(memory, operations_address="10010"),
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
        priority=10,
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
    memory.write(
        str(program_start), 'write("10004", "Hello GPT-4!")'
    )  # Write to GPT-4 input
    memory.write(
        str(program_start + 1), 'trigger_interrupt("syscall")'
    )  # Trigger syscall
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

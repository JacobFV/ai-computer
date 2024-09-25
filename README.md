# AI Computer

LLM = core
processor = group of cores and a shared context and interrupt
function call = micro-op
step-by-step pseudo-code = machine code
specific instructions = assembly
loose instructions = high-level language
notepad = memory
context = the window of memory that the processor is looking at
memory = notepad chunks
reminder = interrupt
computer = processors, memory, interrupts, chats, and peripherals
text 2 text chat that can be reset = peripheral
user chat = user interface (a type of peripheral)
gpt4 chat = a type of peripheral
claude-3.5-sonnet chat = a type of peripheral
task = process
the task to ensure all tasks are completed = kernel
executable = thought graph

## Concepts

- memory contains data
- data can be instructions or just information
- every core has a locations in memory that it is looking at (context)

## Micro-ops

### File System

- read_file(path) -> data
- write_file(path, data) -> void
- move_file(from, to) -> void
- copy_file(from, to) -> void
- delete_file(path) -> void

### Context Management

- find_context(query) -> index
- set_context(index) -> void

### Input/Output (Peripherals)

- input_channel(channel_name) -> text
- output_channel(channel_name, text) -> void
- clear_channel(channel_name) -> void

### Interrupts

- add_interrupt(name, function) -> void
- remove_interrupt(name) -> void
- trigger_interrupt(name) -> void
- list_interrupts() -> list

### Multitasking

- create_process(process_name, process_index) -> process_id
- delete_process(process_id) -> void
- list_processes() -> list
- wait(process_id) -> void = trigger interrupt to set_context(index=kernel_proc_index)
- resume(process_id) -> void = trigger interrupt to set_context(index=process_id)
- kill(process_id) -> void = trigger interrupt to set_context(index=kernel_proc_index)

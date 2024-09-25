# AI Computer

LLM = core
processor = group of cores and a shared context and interrupt
function call = micro-op
step-by-step pseudo-code = assembly
describe in detail prompt = compiler
loose instructions = high-level language
text = memory
context = the window of memory that the processor is looking at
context pointer = the starting index of the context
instruction pointer = the line number that the processor is currently executing
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

- all information is text
- information can come from a peripheral or a core or another internal system but all information is virtualized into a single text space
- information is addressed using line numbers and col numbers. like so: `line:col`. col is optional and if not specified, it is assumed to be 0.
- we have 5 convenience pointer types:
  - exact_match(query, occurrence=0): points to the `occurrence`th occurrence of `query` in the memory space
  - regex_match(regex, occurrence=0): points to the `occurrence`th occurrence of `query` in the memory space
  - tfidf_match(query, k=1): points to the `k` closest lines to `query` in the memory space
  - semantic_match(query, k=1): points to the `k` closest lines to `query` in the memory space
  - meta_pointer(pointer_address): points to another pointer and resolves to whatever that pointer is pointing to
- any micro-op that requests an index can also accept a pointer and the processor will convert it to a line:col address for you
- data can be instructions or just information
- every core has a locations in memory that it is looking at (context)
- the instruction pointer moves down after every micro-op (down instead of up because thats how LLMs were trained to read)
- on each clock cycle, all interrupts are checked and the highest priority interrupt is executed
- interrupts can be triggered manually, on exact match, on regex match, on tfidf match, on semantic match
- an interrupt sets the instruction pointer to a given line and the core will resume execution from there. often the new line will set the context for the entire processor
- the syscall interrupt is usually added to the interrupt table by the compiler but it does not necessarily have to be there

## Micro-ops

### Data Manipulation

- mov(destination_address, source_address) -> void
- write(index, text) -> void
- clear(index) -> void
- insert(insert_index, text) -> void
- delete(index, text) -> void

### Context Management

- find_context(query) -> index
- set_context(index) -> void

### Interrupts

- add_interrupt(name, check_line_no, check_match, goto_line) -> void
- add_interrupt(name, check_line_no, check_regex, goto_line) -> void
- add_interrupt(name, check_line_no, check_tfidf, threshold, goto_line) -> void
- add_interrupt(name, check_line_no, check_query, threshold, goto_line) -> void
- remove_interrupt(name) -> void
- trigger_interrupt(name) -> void
- list_interrupts() -> list

### Other Control Flow

- jump(address) -> void: sets the instruction pointer to the line number specified by `address`
- no_op() -> void (if a process calls no_op 3 times in a row, it will be paused)

## System Calls

The syscall convention is as follows:

1. trigger the syscall interrupt
2. this triggers the interrupt
3. the interrupt sets the instruction pointer to the syscall function in the kernel
4. the syscall function overwrites the syscall line with the return value
5. the syscall function returns and in a high level programming language, the return value is read from the syscall line

### Multitasking

- create_process(process_name, process_index) -> process_id
- delete_process(process_id) -> void
- list_processes() -> list
- pause(process_id) -> void = trigger interrupt to set_context(index=kernel_proc_index)
- resume(process_id) -> void = trigger interrupt to set_context(index=process_id)
- kill(process_id) -> void = trigger interrupt to set_context(index=kernel_proc_index)
- set_priority(process_id, priority) -> void
- get_priority(process_id) -> priority

### File System

- read_file(path) -> text
- write_file(path, text) -> void
- move_file(from, to) -> void
- copy_file(from, to) -> void
- delete_file(path) -> void
- list_files() -> list
- create_file(path) -> void
- delete_file(path) -> void

### Input/Output (Peripherals)

IO channels are virtualized onto memory space, so these are just convenience functions to make it easier to read and write to memory space

- input_channel(channel_name) -> text
- output_channel(channel_name, text) -> void
- clear_channel(channel_name) -> void

### Shell

- run_command(command) -> void
- list_commands() -> list
- get_command_output(command) -> output
- get_command_help(command) -> help

## Memory

- memory is a single line of text
- memory is addressable by line number
- memory can be read from and written to
- memory can be appended to and prepended to
- memory can be deleted
- memory can be replaced

### System memory layout

- line 0: instruction pointer (specific to each core)
- line 1: return value (specific to each core)
- line 2: context pointer (specific to each processor)
- line 3: interrupt table (specific to each processor)
- line 4-1k: reserved for operating system
- line 1k-10k: reserved for kernel
- line 10k-50k: reserved for peripherals
- line 50k-100k: reserved for system processes
- line 100k+: user processes

### Process memory layout

--- Metadata ---
- line 0: instruction pointer
- line 1: context pointer
- line 2: interrupt table
- line 3: executable text segment length
- line 4: structured data segment length
- line 5: unstructured data segment length
- line 6: stack segment length
- line 7-100: reserved for operating system

--- Executable Text Segment ---
one line for each LLM prompt (not including context, which is included at execution time based on the context pointer)

--- Structured Data Segment ---
yaml formatted data
- make dot notation possible
- makes data more organized
- makes it easier to reason about data

--- Unstructured Data Segment ---
raw unstructured text

--- Stack Segment ---
frames are stored here
- each frame is a yaml entry with the following format:
  - uuid: a unique identifier for the frame
  - return_addr: the address to return to after the frame is executed
  - alias (optional): an alias for the frame
  - type (optional): the type of frame (function, loop, condition, etc.)
  - structured data: structured data to append/override to the structured data segment when inside the frame
  - unstructured data: unstructured text to append to the unstructured data segment when inside the frame

## 'Hardware'

- Processor
- Core
- Virtualized Memory
- Peripheral
  - User Interface
  - LLM
  - API
  - Database
  - File System
  - Network

drivers are implementedi n the python runtime that simulates this entire AI processor, and their job is to read/update the apporpriate memory at each time step or when they are triggered

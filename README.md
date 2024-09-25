# AI Computer

## Overview

This project implements an AI Computer architecture in Python, utilizing language models for instruction parsing and execution. It simulates traditional computing components such as memory, processors, cores, and peripherals, reimagined in the context of AI and large language models.

## Features

- **Memory**: Virtualized memory space with line and column addressing.
- **Processors and Cores**: Multi-core processor simulation.
- **Micro-operations**: Basic operations for memory manipulation and control flow.
- **Interrupt Handling**: Supports interrupts with priority management.
- **Peripherals**: Includes tick counter, clock, user chat, GPT-4 chat, bash shell, and filesystem driver.
- **Language Model Integration**: Uses `ell` and OpenAI\'s GPT-4 for instruction parsing and execution.
- **Pointer Functions**: Supports `exact_match`, `regex_match`, `tfidf_match`, and `semantic_match` for memory addressing.

## Project Structure

```

ai_computer/
├── main.py
├── memory.py
├── processor.py
├── micro_ops.py
├── interrupts.py
├── peripherals/
│   ├── **init**.py
│   ├── base.py
│   ├── tick_counter.py
│   ├── clock.py
│   ├── user_chat.py
│   ├── gpt4_chat.py
│   ├── bash_shell.py
│   └── filesystem.py
├── requirements.txt
└── README.md

```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/JacobFV/ai_computer.git
   cd ai_computer
   ```

2. **Install dependencies**:

   ```bash
   poetry install
   ```

   - If you don't have Poetry installed, you can install it with:

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Activate the virtual environment**:

   ```bash
   poetry shell
   ```

4. **Set up OpenAI API key**:

   Ensure you have an OpenAI API key and set it as an environment variable:

   ```bash
   export OPENAI_API_KEY="sk-proj-1234****"
   ```

## Usage

Run the main program:

```bash
python main.py
```

## Notes

- **Performance Considerations**: Some operations, especially semantic matching, can be computationally intensive. Performance optimizations such as caching are implemented, but further improvements may be needed.
- **Security**: This is a conceptual implementation and may not include all necessary security measures. Use with caution.
- **Dependencies**: Ensure all dependencies are installed and compatible with your system.

## Future Work

- **Expand Pointer Functions**: Complete implementations for all pointer functions.
- **Enhance Error Handling**: Improve error detection and handling throughout the system.
- **Security Enhancements**: Implement robust security measures for process isolation and memory access control.
- **User Interface Improvements**: Develop a GUI or web interface for better interaction.
- **Testing and Validation**: Create unit tests and perform thorough testing.

## License

[MIT License](./LICENSE)

## Acknowledgments

- Thanks to the open-source community for providing the tools and libraries used in this project.

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
   git clone https://github.com/yourusername/ai_computer.git
   cd ai_computer
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API key**:

   Ensure you have an OpenAI API key and set it as an environment variable:

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
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

[MIT License](LICENSE)

## Acknowledgments

- Thanks to the open-source community for providing the tools and libraries used in this project.
'

# Create README.md

# Create LICENSE (optional, since README references it)

create_file "LICENSE" '
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
...
[Full MIT License text here]
'

# Make the script executable (optional)

chmod +x generate_ai_computer.sh

echo "AI Computer codebase has been generated successfully in the '$PROJECT_ROOT' directory."

```

---

**Explanation of the Script:**

1. **Shebang and Safety Measures:**
   - `#!/bin/bash`: Specifies that the script should be run in the Bash shell.
   - `set -e`: Ensures the script exits immediately if any command fails, preventing partial setups.

2. **Project Directory Creation:**
   - Defines `PROJECT_ROOT` as `ai_computer`.
   - Creates the main project directory and navigates into it.
   - Creates the `peripherals` subdirectory.

3. **File Creation Function (`create_file`):**
   - A helper function that takes a file path and content, creating the file with the specified content using a `here-document` (`cat <<EOF > "$filepath"`).

4. **File Population:**
   - **`memory.py`**: Implements the `Memory` class with line and column addressing, pointer resolution, and semantic matching using BERT embeddings.
   - **`micro_ops.py`**: Defines micro-operations with a decorator for clean registration.
   - **`interrupts.py`**: Manages interrupts with priorities and conditions.
   - **`processor.py`**: Defines the `Processor` and `Core` classes, handling instruction execution using the `ell` library for LLM interactions.
   - **Peripherals**:
     - **`base.py`**: Abstract base class for all peripherals.
     - **`tick_counter.py`**: Increments a tick count every second.
     - **`clock.py`**: Updates the current time every second.
     - **`user_chat.py`**: Handles asynchronous user input and output.
     - **`gpt4_chat.py`**: Interfaces with GPT-4 using the `ell` library.
     - **`bash_shell.py`**: Executes shell commands and returns outputs.
     - **`filesystem.py`**: Manages file system operations based on memory instructions.
     - **`__init__.py`**: Initializes the `peripherals` package.
   - **`main.py`**: Sets up the memory, processor, peripherals, interrupts, and loads a simple program into memory for execution.
   - **`requirements.txt`**: Lists all necessary Python dependencies.
   - **`README.md`**: Provides an overview, installation instructions, usage guide, and other relevant information.
   - **`LICENSE`**: Placeholder for the MIT License (you should replace `[Full MIT License text here]` with the actual license text).

5. **Final Steps:**
   - Makes the script executable (optional).
   - Prints a success message upon completion.

---

**Usage Instructions:**

1. **Save the Script:**
   - Save the provided script content into a file named `generate_ai_computer.sh`.

2. **Run the Script:**
   - Open your terminal.
   - Navigate to the directory where `generate_ai_computer.sh` is saved.
   - Make the script executable (optional):
     ```bash
     chmod +x generate_ai_computer.sh
     ```
   - Execute the script:
     ```bash
     ./generate_ai_computer.sh
     ```
     or
     ```bash
     bash generate_ai_computer.sh
     ```

3. **Post-Execution:**
   - The script will create an `ai_computer` directory with all the necessary files and directories.
   - Navigate into the project directory:
     ```bash
     cd ai_computer
     ```
   - **Set Up Virtual Environment (Recommended):**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Install Dependencies:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Set OpenAI API Key:**
     ```bash
     export OPENAI_API_KEY="your-api-key-here"
     ```
   - **Run the AI Computer:**
     ```bash
     python main.py
     ```

**Note:**
- Ensure you have Python 3.7 or higher installed.
- Replace `"your-api-key-here"` with your actual OpenAI API key.
- The `LICENSE` file in the script is a placeholder. You should insert the full MIT License text or your preferred license.

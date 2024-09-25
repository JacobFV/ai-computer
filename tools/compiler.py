from pathlib import Path

import typer
import ell

app = typer.Typer()
app_dir = Path(__file__).parent

# Initialize ell with versioning and tracing
ell.init(store="./logdir", autocommit=True, verbose=True)


@app.command()
def compile(path: str, output: str):
    """
    Compile AI computer files and generate a detailed description.
    """
    # Read the input file
    with open(path, "r") as file:
        input = file.read()

    # Read and concatenate all .py and .md files in ./ai_computer/
    concatenated_code = ""
    for file_path in Path(app_dir / "ai_computer").rglob("*"):
        if file_path.suffix in [".py", ".md"]:
            with open(file_path, "r") as file:
                concatenated_code += f"<{file_path}>\n{file.read()}\n</{file_path}>\n\n"

    # Append the input to the concatenated code
    full_input = f"{concatenated_code}\n{input}"

    # Use ell to generate the description with @ell.complex
    @ell.complex(model="gpt-4")
    def generate_description(code_and_input: str):
        """You are an AI expert specializing in AI computer systems. Given the code and documentation of an AI system and an input, generate an extremely detailed, ultra-literal description of what the AI should do. Include insights into the system architecture, processes, and any potential edge cases."""
        return [
            ell.system("You are an AI expert specializing in AI computer systems."),
            ell.user(
                f"Based on the following code and input, provide an extremely detailed and ultra-literal description of what the AI should do:\n\n{code_and_input}"
            ),
        ]

    description_message = generate_description(full_input)

    # Write the description to the output file
    with open(output, "w") as file:
        file.write(description_message.text)


if __name__ == "__main__":
    app()

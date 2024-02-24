import subprocess

def run_command(command: str) -> str:
    """
    Executes a shell command and returns the output.

    Args:
    - command (str): The command to run.

    Returns:
    - str: The output of the command.
    """
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    if result.stderr:
        raise Exception(result.stderr.decode())
    return result.stdout.decode()

def process_pdf_in_container(container_name: str, local_pdf_path: str) -> str:
    """
    Processes a PDF file by copying it into a Docker container and running a Python script on it.

    Args:
    - container_name (str): The name of the Docker container.
    - local_pdf_path (str): The local path to the PDF file.

    Returns:
    - str: The output from processing the PDF.
    """
    copy_command = f"docker cp {local_pdf_path} {container_name}:/tmp"
    run_command(copy_command)

    python_command = f"docker exec {container_name} python3 -c \"from unstructured.partition.pdf import partition_pdf; elements = partition_pdf(filename='/tmp/{local_pdf_path.split('/')[-1]}'); print(elements)\""
    output = run_command(python_command)

    return output

def process_text_in_container(container_name: str, local_text_path: str) -> str:
    """
    Processes a text file by copying it into a Docker container and running a Python script on it.

    Args:
    - container_name (str): The name of the Docker container.
    - local_text_path (str): The local path to the text file.

    Returns:
    - str: The output from processing the text file.
    """
    copy_command = f"docker cp {local_text_path} {container_name}:/tmp"
    run_command(copy_command)

    python_command = f"docker exec {container_name} python3 -c \"from unstructured.partition.text import partition_text; elements = partition_text(filename='/tmp/{local_text_path.split('/')[-1]}'); print(elements)\""
    output = run_command(python_command)

    return output

def run_script_in_container(container_name: str, script_path: str) -> str:
    """
    Executes a custom Python script within a Docker container.

    Args:
    - container_name (str): The name of the Docker container.
    - script_path (str): The local path to the Python script.

    Returns:
    - str: The output from executing the script.
    """
    copy_command = f"docker cp {script_path} {container_name}:/tmp/script_to_run.py"
    run_command(copy_command)

    exec_command = f"docker exec {container_name} python3 /tmp/script_to_run.py"
    output = run_command(exec_command)

    return output

# # Example usage
# container_name = "unstructured"
# pdf_path = "example-docs/layout-parser-paper-fast.pdf"
# text_path = "example-docs/fake-text.txt"

# # Process a PDF file
# pdf_output = process_pdf_in_container(container_name, pdf_path)
# print(pdf_output)

# # Process a text file
# text_output = process_text_in_container(container_name, text_path)
# print(text_output)

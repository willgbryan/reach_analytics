import subprocess
import json

def run_command(command: str) -> str:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    if result.stderr:
        raise Exception(result.stderr.decode())
    return result.stdout.decode()

def process_pdf_in_container(container_name: str, local_pdf_path: str) -> str:
    copy_command = f"docker cp {local_pdf_path} {container_name}:/tmp"
    run_command(copy_command)

    python_command = f"docker exec {container_name} python3 -c \"from unstructured.partition.pdf import partition_pdf; elements = partition_pdf(filename='/tmp/{local_pdf_path.split('/')[-1]}'); print(elements)\""
    output = run_command(python_command)

    return output

def process_text_in_container(container_name: str, local_text_path: str) -> str:
    copy_command = f"docker cp {local_text_path} {container_name}:/tmp"
    run_command(copy_command)

    python_command = f"docker exec {container_name} python3 -c \"from unstructured.partition.text import partition_text; elements = partition_text(filename='/tmp/{local_text_path.split('/')[-1]}'); print(elements)\""
    output = run_command(python_command)

    return output

# Example usage
container_name = "unstructured"
pdf_path = "example-docs/layout-parser-paper-fast.pdf"
text_path = "example-docs/fake-text.txt"

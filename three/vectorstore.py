import marqo
import subprocess

# run this to host the marqo db

def spin_up_local_vdb() -> marqo.Client:
    commands = [
    "docker pull marqoai/marqo:latest",
    "docker rm -f marqo",
    "docker run --name marqo -it --privileged -p 8882:8882 --add-host host.docker.internal:host-gateway marqoai/marqo:latest"
    ]

    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()

        if process.returncode != 0:
            print(f"Error executing command: {cmd}")
            print(err.decode())
        else:
            print(out.decode())

    return marqo.Client(url="http://localhost:8882")


mq = spin_up_local_vdb()
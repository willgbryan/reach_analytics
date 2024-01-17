import io
import docker

def check_for_image(image_name: str = 'docker-iso-runtime') -> bool:
    client = docker.from_env()
    try:
        client.images.get(image_name)
        return True
    except docker.errors.ImageNotFound:
        return False

def build_docker_image(image_name: str = 'docker-iso-runtime'):
    if check_for_image(image_name):
        print(f"Docker image {image_name} already exists. Skipping build.")
        return

    client = docker.from_env()
    with open("Dockerfile", "r") as f:
        file_content = f.read()
    
    image, build_logs = client.images.build(
        fileobj=io.BytesIO(file_content.encode('utf-8')),
        tag=image_name,
        rm=True
    )
    for log in build_logs:
        print(log.get("stream", "").strip())

def docker_runtime(code: str, image_name: str = 'docker-iso-runtime') -> str:
    client = docker.from_env()

    container = client.containers.run(
        image_name, 
        command=f"python -c '{code}'", 
        remove=True, 
        detach=True
    )

    response = container.wait()
    logs = container.logs().decode('utf-8')
    
    if response['StatusCode'] != 0:
        raise Exception(logs)
    
    return logs

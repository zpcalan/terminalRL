import docker, tempfile, re
import os,time
def create_image_and_run(index, docker_cmd, model_cmd, python_script):
    global image_list
    docker_client = docker.from_env()
    image_name = f"image_{str(index)}"
    has_image = False
    try:
        image = docker_client.images.get(image_name)
        has_image = True
    except docker.errors.ImageNotFound:
        print(f"image for {image_name} not found")
        has_image=False
    if not has_image:
        temp_dir = tempfile.mkdtemp()
        dockerfile_path = os.path.join(temp_dir, "Dockerfile")
        with open(dockerfile_path, 'w') as f:
            f.write(docker_cmd)
            
        image_name = f"image_{str(index)}"
        image, build_logs = docker_client.images.build(path=temp_dir, tag=image_name, rm=True)
        print(f"build image for docker file {docker_cmd} log: {build_logs}")
        image_list.append(image_name)

    try:
        container = docker_client.containers.run(
                image_name,
                command= model_cmd,
                detach=False,
                stdout=True,
                stderr=True,
                remove=True
            )
    #except Exception as e:
    except docker.errors.ContainerError as e:
        print(f"ERROR: {str(e)}")
    logs = container.decode('utf-8')
    print(f"{model_cmd}logs are {logs}")
    return logs

def terminal_reward_fn(index, result, docker_cmd, test_weights, python_script):
    # Extract command lines from the completion.
    pattern = r'<cmd>(.*?)</cmd>'
    matches = re.findall(pattern, result, re.DOTALL)
    result = create_image_and_run(index, docker_cmd, matches[0], python_script)

    # TODO: Consider test_weights to construct smooth reward output.
    if "success".lower() in result.lower():
        return 1, result
    else:
        return 0, result


result = "<cmd>python --version && mpirun --version</cmd>"
docker_cmd = "FROM python:3.9-slim"
test_python_script = """
import time
print(f"zpc time is {time.time()}")
"""
terminal_reward_fn(0, result, docker_cmd, "", test_python_script)

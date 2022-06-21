import glob
import os
import shutil
import time
import torch
from pathlib import Path
join = os.path.join
from logger import add_file_handler_to_logger, logger


def check_dir(file_path):
    file_path = Path(file_path)
    files = [f for f in file_path.iterdir() if ".nii.gz" in str(f)]
    if len(files) != 0:
        return False
    return True


add_file_handler_to_logger(name="main", dir_path="logs/", level="DEBUG")

docker_path = './team_docker'
test_img_path = './ValidationImg/'
save_path = './results/'

os.makedirs(save_path, exist_ok=True)
os.makedirs('./inputs/', exist_ok=True)
os.makedirs('./outputs/', exist_ok=True)

dockers = sorted(os.listdir(docker_path))
test_cases = sorted(os.listdir(test_img_path))

for docker in dockers:
    try:
        name = docker.split('.')[0].lower()
        print('teamname docker: ', docker)
        os.system('docker image load < {}'.format(join(docker_path, docker)))
        team_outpath = join(save_path, name)
        if os.path.exists(team_outpath):
            shutil.rmtree(team_outpath)
        os.mkdir(team_outpath)
        for case in test_cases:
            if not check_dir('./inputs'):
                logger.error("please check inputs folder")
                raise
            shutil.copy(join(test_img_path, case), './inputs')
            start_time = time.time()
            os.system('python Efficiency.py -docker_name {}'.format(name))
            logger.info(f"{case} finished!")
            os.remove(join('./inputs', case))
            # shutil.rmtree('./inputs')
            logger.info(f"{case} cost time: {time.time() - start_time}")

        os.system("python load_json.py -docker_name {} -save_path {}".format(name, save_path))
        shutil.move("./outputs", team_outpath)
        os.mkdir("./outputs")
        torch.cuda.empty_cache()
        os.system("docker rmi {}:latest".format(name))
    except Exception as e:
        logger.exception(e)

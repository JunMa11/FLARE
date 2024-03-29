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

docker_path = './team_docker/' # put docker in this folder
test_img_path =  './data/validation' # all validation cases
save_path = './FLARE23_ValResults/' # evaluation results will be saved in this folder

temp_in = './inputs/'
temp_out = './outputs'
os.makedirs(save_path, exist_ok=True)
os.makedirs(temp_in, exist_ok=True)
os.makedirs(temp_out, exist_ok=True)
os.system("chmod -R 777 outputs/")

dockers = sorted(os.listdir(docker_path))
test_cases = sorted(os.listdir(test_img_path))

for docker in dockers:
    try:
        name = docker.split('.')[0].lower()
        print('loading docker:', docker)
        os.system('docker load -i {}'.format(join(docker_path, docker)))
        team_outpath = join(save_path, name)
        if os.path.exists(team_outpath):
            shutil.rmtree(team_outpath)
        os.mkdir(team_outpath)
        for case in test_cases:
            if not check_dir(temp_in):
                logger.error("please check inputs folder", temp_in)
                raise
            shutil.copy(join(test_img_path, case), temp_in)
            start_time = time.time()
            try:
                shutil.rmtree('./inputs/nnUNet_raw/')
                shutil.rmtree('./inputs/nnUNet_preprocessed/')
            except:
                print('no temp files')
            os.system('python Efficiency.py -docker_name {} -save_file {}'.format(name, save_path))
            logger.info(f"{case} finished!")
            os.remove(join('./inputs', case))
            try:
                shutil.rmtree('./inputs/nnUNet_cropped_data/')
                shutil.rmtree('./inputs/nnUNet_raw_data/')
                shutil.rmtree('./inputs/nnUNet_preprocessed/')
            except:
                print('no temp files')
            # shutil.rmtree('./inputs')
            logger.info(f"{case} cost time: {time.time() - start_time}")
            # move segmentation file
            seg_name = case.split('_0000.nii.gz')[0]+'.nii.gz'
            if os.path.exists(join(temp_out, seg_name)):
                os.rename(join(temp_out, seg_name), join(team_outpath, seg_name))

        os.system("python load_json.py -docker_name {} -save_path {}".format(name, save_path))
        shutil.move(temp_out, team_outpath)
        os.mkdir(temp_out)
        torch.cuda.empty_cache()
        shutil.rmtree(temp_in)
        os.mkdir(temp_in)
        os.system("docker rmi {}:latest".format(name))
    except Exception as e:
        logger.exception(e)

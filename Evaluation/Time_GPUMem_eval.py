import shutil,os
import glob
import torch

tarlist = glob.glob('./team_docker/*.tar')
for item in tarlist:
    name = item.split('/')[-1].split('.')[0].lower()
    #print(name )
    os.system('docker image load < {}'.format(item))
    os.mkdir('./data_all/{}'.format(name))
    for image_i in  sorted(glob.glob('imagesTs/*')):
        shutil.copy(image_i,image_i.replace('imagesTs','inputs'))
        os.system('python Efficiency.py -docker_name {}'.format(name))
        print(image_i.split('/')[-1],'finished!')
        os.remove(image_i.replace('imagesTs','inputs'))
    os.system('python load_json.py -docker_name {}'.format(name))
    shutil.move('./outputs','./data_all/{}/'.format(name))
    os.mkdir('./outputs')
    torch.cuda.empty_cache()

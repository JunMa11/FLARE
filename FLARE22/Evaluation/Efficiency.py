import argparse
import glob
import json
import os
import time
import psutil as p
from multiprocessing import Manager, Process
from pynvml.smi import nvidia_smi
import numpy as np
from logger import add_file_handler_to_logger, logger

add_file_handler_to_logger(name="main", dir_path="logs/", level="DEBUG")

manager = Manager()
gpu_list = manager.list()
cpu_list = manager.list()
RAM_list = manager.list()
def cpu_usage():
    t = p.cpu_times()
    return [t.user, t.system, t.idle]
before = cpu_usage()
def get_cpu_usage():
    global before
    now = cpu_usage()
    delta = [now[i] - before[i] for i in range(len(now))]
    total = sum(delta)
    before = now
    return [(100.0*dt)/(total+0.1) for dt in delta]


def daemon_process(time_interval, json_path, gpu_index=1):
    while True:
        nvsmi = nvidia_smi.getInstance()
        dictm = nvsmi.DeviceQuery("memory.free, memory.total")
        gpu_memory = (
            dictm["gpu"][gpu_index]["fb_memory_usage"]["total"] - dictm["gpu"][gpu_index]["fb_memory_usage"]["free"]
        )
        cpu_usage =get_cpu_usage()   
        RAM = p.virtual_memory().used/1048576
        gpu_list.append(gpu_memory)
        RAM_list.append(RAM)
        cpu_list.append(cpu_usage)
        # with open(json_path, 'w')as f:
        #     #js['gpu_memory'] = gpu_memory_max
        #     js['gpu_memory'].append(gpu_memory)
        #     json.dump(js, f, indent=4)
        time.sleep(time_interval)


def save_result(start_time, sleep_time, json_path, gpu_list_ ,cpu_list_copy,RAM_list_copy):
    if os.path.exists(json_path):
        with open(json_path) as f:
            js = json.load(f)
    else:
        js = {"gpu_memory": []}
        # raise ValueError(f"{json_path} don't exist!")

    with open(json_path, "w") as f:
        # js['gpu_memory'] = gpu_memory_max
        js["gpu_memory"] = gpu_list_
        js["cpu_list"] = cpu_list_copy 
        js["RAM_list"] = RAM_list_copy 
        
        json.dump(js, f, indent=4)

    infer_time = time.time() - start_time
    with open(json_path, "r") as f:
        js = json.load(f)
    with open(json_path, "w") as f:
        js["time"] = infer_time
        json.dump(js, f, indent=4)
    time.sleep(2)
    logger.info("save result end")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-time_interval", default=0.1, help="time_interval")
    parser.add_argument("-sleep_time", default=5, help="sleep time")
    parser.add_argument("-shell_path", default="predict.sh", help="time_interval")
    # XXX: in case of someone use lower docker, please use specified GPU !!!
    parser.add_argument("-gpus", default=1, help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("-docker_input_file", default="./inputs/", help="docker input folder")
    parser.add_argument("-save_file", default="results", help="data save folder")
    parser.add_argument("-docker_name", default="nnunet", help="docker output folder")
    args = parser.parse_args()
    logger.info(f"We are evaluating {args.docker_name}")
    json_dir = "./{}/{}".format(args.save_file,args.docker_name)
    json_path = os.path.join(
        json_dir, glob.glob(args.docker_input_file + "/*")[0].split("/")[-1].split(".")[0] + ".json",
    )

    try:
        p1 = Process(target=daemon_process, args=(args.time_interval, json_path, args.gpus,))
        p1.daemon = True
        p1.start()
        t0 = time.time()
        # XXX: in case of someone use lower docker, please use specified GPU !!!
        # cmd = 'docker container run --runtime="nvidia"  -e NVIDIA_VISIBLE_DEVICES={} --name {} --rm -v $PWD/inputs/:/workspace/input/ -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ {}:latest /bin/sh -c "sh {}"'.format(
        #     args.gpus, args.docker_name, args.docker_name, args.shell_path
        # )
        cmd = 'docker container run --gpus="device=1" --name {} --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ {}:latest /bin/bash -c "sh predict.sh" '.format(args.docker_name, args.docker_name)
        logger.info(f"cmd is : {cmd}")
        logger.info(f"cmd is : {cmd}")
        logger.info("start predict...")
        os.system(cmd)
        RAM_list= list(RAM_list)
        RAM_list_copy = RAM_list.copy()
        cpu_list= list(cpu_list)
        cpu_list_copy = cpu_list.copy()
        gpu_list = list(gpu_list)
        gpu_list_copy = gpu_list.copy()
        save_result(t0, args.sleep_time, json_path, gpu_list_copy ,cpu_list_copy,RAM_list_copy)
        time.sleep(args.sleep_time)
    except Exception as error:
        logger.exception(error)

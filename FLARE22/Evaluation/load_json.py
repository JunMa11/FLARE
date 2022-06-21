import json
import csv
import argparse
import glob
import matplotlib
import os
join = os.path.join
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from logger import add_file_handler_to_logger, logger

add_file_handler_to_logger(name="main", dir_path=f"logs/", level="DEBUG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-docker_name", default="fully_suplearn_subtask1", help="team docker name")
    parser.add_argument("-save_path", default="./results", help="save_path")
    time_interval = 0.1
    args = parser.parse_args()
    logger.info("we are counting: {args.docker_name}")
    json_dir = join(args.save_path, args.docker_name)
    csv_path = join(json_dir, args.docker_name + '_Efficiency.csv')
    jsonl = sorted(glob.glob(json_dir + "/*.json"))
    alldata = []
    for item in jsonl:
        csv_l = []
        name = item.split(os.sep)[-1].split(".")[0]
        csv_l.append(name + '.nii.gz')
        zitem = item
        with open(zitem) as f:
            try:
                js = json.load(f)
            except Exception as error:
                logger.error(f"{item} have error")
                logger.exception(error)
            if "time" not in js:
                logger.error(f"{item} don't have time!!!!")
                logger.info(f"Manually compute {item}")
                time = time_interval * len(js["gpu_memory"])
            else:
                time = js["time"]
            csv_l.append(np.round(time,2))
            #CPU
            user, system, all_cpu_used = [item[0] for item in js['cpu_list']], [item[1] for item in js['cpu_list']], [
                100 - item[2] for item in js['cpu_list']]
            plt.cla()
            x = [item * time_interval for item in range(len(user))]
            plt.xlabel("Time (s)", fontsize="large")
            plt.ylabel("CPU Utlization (%)", fontsize="large")
            # plt.plot(x, user, "b", ms=10, label="User %")
            # plt.plot(x, system, "r", ms=10, label="System %")
            plt.plot(x, all_cpu_used, "b", ms=10, label="Used %")
            plt.legend()
            plt.savefig(zitem.replace(".json", "_CPU-Time.png"))
            #RAM
            RAM = js["RAM_list"]
            plt.cla()
            x = [item * time_interval for item in range(len(RAM))]
            plt.xlabel("Time (s)", fontsize="large")
            plt.ylabel("RAM (MB)", fontsize="large")
            plt.plot(x, RAM, "b", ms=10, label="RAM")
            plt.legend()
            plt.savefig(zitem.replace(".json", "_RAM-Time.png"))
            mem = js["gpu_memory"]
            x = [item * time_interval for item in range(len(mem))]
            plt.cla()
            plt.xlabel("Time (s)", fontsize="large")
            plt.ylabel("GPU Memory (MB)", fontsize="large")
            plt.plot(x, mem, "b", ms=10, label="a")
            plt.savefig(zitem.replace(".json", "_GPU-Time.png"))
            count_set = set(mem)
            count_list = []
            for citem in count_set:
                cts = mem.count(citem)
                if cts > 0.02 * len(mem):
                    count_list.append(citem)
            max_mem = max(count_list)
            csv_l.append(np.round(max_mem))
            csv_l.append(np.round(sum(mem) * time_interval))
            csv_l.append(np.round(max(all_cpu_used),2))
            csv_l.append(np.round(sum(all_cpu_used) * time_interval,2))
            csv_l.append(np.round(max(RAM),2))
            csv_l.append(np.round(sum(RAM) * time_interval))
        alldata.append(csv_l)

    f = open(csv_path, "w",newline='')
    writer = csv.writer(f)
    writer.writerow(["Name", "Time", "GPU_Mem", "AUC_GPU_Time",'CPU_Utilization','AUC_CPU_Time','RAM'
                        ,'AUC_RAM_Time'    ])
    for i in alldata:
        writer.writerow(i)
    f.close()

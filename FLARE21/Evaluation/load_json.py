import json
import csv
import argparse
import glob
import matplotlib
import os
join = os.path.join

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from logger import add_file_handler_to_logger, logger

add_file_handler_to_logger(name="main", dir_path=f"logs/", level="DEBUG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-docker_name", default="fully_suplearn_subtask1", help="team docker name")
    parser.add_argument("-save_path", default="./results", help="save_path")
    args = parser.parse_args()
    logger.info("we are counting: {args.docker_name}")
    json_dir = join(args.save_path, args.docker_name)
    csv_path = join(json_dir, args.docker_name + '_Efficiency.csv')
    jsonl = sorted(glob.glob(json_dir + "/*.json"))
    alldata = []
    for item in jsonl:
        csv_l = []
        name = item.split("/")[-1].split(".")[0]
        csv_l.append(name)
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
                time = 0.1 * len(js["gpu_memory"])
            else:
                time = js["time"]
            csv_l.append(time)
            mem = js["gpu_memory"]
            x = [item * 0.1 for item in range(len(mem))]
            plt.cla()
            plt.xlabel("Time (s)", fontsize="large")
            plt.ylabel("GPU Memory (MB)", fontsize="large")
            plt.plot(x, mem, "b", ms=10, label="a")
            plt.savefig(zitem.replace(".json", ".jpg"))
            count_set = set(mem)
            count_list = []
            for citem in count_set:
                cts = mem.count(citem)
                if cts > 0.02 * len(mem):
                    count_list.append(citem)
            max_mem = max(count_list)
            csv_l.append(max_mem)
        csv_l.append(sum(mem) * 0.1)
        alldata.append(csv_l)
    f = open(csv_path, "w")
    writer = csv.writer(f)
    writer.writerow(["name", "time", "gpu_memory", "time_multiply_memory"])
    for i in alldata:
        writer.writerow(i)
    f.close()

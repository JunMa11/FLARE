# Official repository of MICCAI 2021 Challenge: [FLARE21](https://flare.grand-challenge.org/FLARE21/).

We provide the evaluation code and a [video demo](https://www.bilibili.com/video/BV1mU4y1n7V6) of the evaluation process. You can use the code to evaluate the `Running time` and `Maximum used GPU memory` of your Docker.

Please feel free to raise any issues if you have questions about the challenge, e.g., dataset, evaluation measures, ranking scheme and so on.

### Enviroment and requirement

- python 3.8+
- torch
- loguru
- pynvml


### Compute running time and GPU memory

Set `docker_path`, `test_img_path`, and `save_path` in `Time_GPUMem_eval.py` and run

`nohup python Time_GPUMem_eval.py >> infos.log &`

### Compute DSC and NSD

Set `seg_path`, `gt_path`, `save_path`, `save_name` in `DSC_NSD_eval.py` and run

`python DSC_NSD_eval.py`


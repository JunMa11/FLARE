# Segmentation Efficiency Evaluation



### Enviroment and requirements

```python
- python 3.8+
- torch
- loguru
- pynvml
- psutil
```



Set `docker_path`, `test_img_path`, and `save_path` in `resource_eval.py` and run

`nohup python resource_eval.py >> infos.log &`



Q: How the `Area under GPU memory-time curve` and `Area under CPU utilization-time curve` is computed?

> A: We record the GPU memory and GPU utilization every 0.1s. The `Area under GPU memory-time curve` and `Area under CPU utilization-time curve` are the cumulative values along running time.



Note: For DSC and NSD, please refer to `FLARE22_DSC_NSD_Eval.py`.




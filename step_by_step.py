# package import
from data_pack import DataSetDefinition
import utils_cpu
from utils_gpu import elem_generate_hint

from typing import List, Dict
from tqdm import tqdm

import ray
import os
import argparse

from functools import partial


def directory_composer(cli_arg):

    resolution_4x = cli_arg.resolution_4x
    ray.init()

    dataset_def = DataSetDefinition(
        processed_dir_suffix=cli_arg.processed_dir_suffix,
        resolution_4x=cli_arg.resolution_4x,
    )

    if resolution_4x:
        dataset_root_dir = os.path.join(dataset_def.processed_data_root_dir, "4x")
    else:
        dataset_root_dir = dataset_def.processed_data_root_dir

    wrapped_process_image = ray.remote(num_cpus=4)(utils_cpu.process_image)

    ### train ###
    processed_dir = os.path.join(dataset_root_dir, "train")
    ray_pack = []
    print("Processing train data")
    for each_elem in tqdm(dataset_def.step1_datadict_train_list):
        if resolution_4x:
            ray_pack.append(
                wrapped_process_image.remote(
                    data_dict=each_elem, output_dir=processed_dir, resoultion=4.0
                )
            )
        else:
            ray_pack.append(
                wrapped_process_image.remote(
                    data_dict=each_elem, output_dir=processed_dir
                )
            )
    result = ray.get(ray_pack)

    ### eval ###
    processed_dir = os.path.join(dataset_root_dir, "eval")
    ray_pack = []
    print("Processing eval data")
    ray_pack = []
    for each_elem in tqdm(dataset_def.step1_datadict_eval_list):
        if resolution_4x:
            ray_pack.append(
                wrapped_process_image.remote(
                    data_dict=each_elem, output_dir=processed_dir, resoultion=4.0
                )
            )
        else:
            ray_pack.append(
                wrapped_process_image.remote(
                    data_dict=each_elem, output_dir=processed_dir
                )
            )
    result = ray.get(ray_pack)

    print("processing done")


def radiance_compute(cli_arg):
    ray.init()

    wrapped_elem_generate_hint = ray.remote(num_gpus=cli_arg.num_gpus)(
        elem_generate_hint
    )

    dataset_def = DataSetDefinition(
        processed_dir_suffix=cli_arg.processed_dir_suffix,
        resolution_4x=cli_arg.resolution_4x,
    )

    step2_dict = dataset_def.step2_dict_generator(resolution_4x=cli_arg.resolution_4x)

    ray_pack = []

    for key, val in step2_dict.items():
        print(f"Processing {key} data")
        for each_elem in tqdm(val):
            ray_pack.append(wrapped_elem_generate_hint.remote(args=each_elem))

    result = ray.get(ray_pack)


if __name__ == "__main__":
    cli_arg = argparse.ArgumentParser()
    cli_arg.add_argument("--process_target", type=str, default="directory_composer")
    cli_arg.add_argument("--processed_dir_suffix", type=str, default=None)
    cli_arg.add_argument("--resolution_4x", action="store_true")
    cli_arg.add_argument("--num_gpus", type=float, default=1.0)

    cli_arg = cli_arg.parse_args()

    if cli_arg.process_target == "directory_composer":
        directory_composer(cli_arg)
    elif cli_arg.process_target == "radiance_compute":
        radiance_compute(cli_arg)
    else:
        raise ValueError("Invalid process_target")

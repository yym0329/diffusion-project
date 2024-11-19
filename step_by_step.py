# package import
from data_pack import DataSetDefinition
import utils_cpu

from typing import List, Dict
from tqdm import tqdm

import ray
import os

def main():
    ray.init()
    
    wrapped_process_image = ray.remote(num_cpus=1)(utils_cpu.process_image)
    
    dataset_def = DataSetDefinition()
    
    processed_dir = os.path.join(dataset_def.processed_data_root_dir,
                                 "train")
    ray_pack = []
    print("Processing train data")
    for each_elem in tqdm(dataset_def.step1_datadict_train_list):
        ray_pack.append(wrapped_process_image.remote(
            data_dict=each_elem,
            output_dir=processed_dir
            )) 
    
    result = ray.get(ray_pack)
    print("processing done")

if __name__ == "__main__":
    main()
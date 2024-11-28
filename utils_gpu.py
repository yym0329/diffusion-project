# 영민님 code base를 parallel하게 돌리기 위함
import json
import numpy as np
from PIL import Image
import random
import os
import glob
import argparse
from dataclasses import dataclass
from typing import Optional
import imageio
import numpy as np
import cv2
import simple_parsing
from transformers import pipeline
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch


from easydict import EasyDict

from typing import Dict

# internal Libs
from DiLightNet.demo.mesh_recon import mesh_reconstruction  # depth to mesh
from DiLightNet.demo.render_hints import (
    render_hint_images,
)  # mesh, env_map -> radiance hints
from DiLightNet.demo.rm_bg import rm_bg


# caption generator
class CaptionGenerator:
    def __init__(self, device: str = "cuda:0"):
        # Use a pipeline as a high-level helper

        self.pipe = pipeline("image-to-text", model="Salesforce/blip2-opt-2.7b", device=device)

    def __call__(self, img_path):
        image = Image.open(img_path).convert("RGB")
        caption = self.pipe(image)
        return caption[0]["generated_text"]

class BLIPI2T:
    def __init__(self):
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to("cuda:0")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        # self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map="auto")
        # self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        
    def __call__(self, img_path):
        image = Image.open(img_path) # .convert("RGB")
        
        # import ipdb; ipdb.set_trace()
        inputs = self.processor(images=image, return_tensors="pt").to("cuda", torch.float16)
        caption = self.model.generate(**inputs)
        caption = self.processor.decode(caption[0], skip_special_tokens=True)
        return caption


class BLIPI2TLarge:
    # /data2/code/diffusion-project/weight/blip2-flan-t5-xl
    def __init__(self):
        # self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to("cuda:0")
        # self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        # /data2/code/diffusion-project/weight/blip2-flan-t5-xl
        
        self.model = Blip2ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path="Salesforce/blip2-flan-t5-xl", cache_dir="/data2/code/diffusion-project/weight/blip2-flan-t5-xl", load_in_8bit=True, device_map="auto")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        
    def __call__(self, img_path):
        image = Image.open(img_path) # .convert("RGB")
        
        # import ipdb; ipdb.set_trace()
        inputs = self.processor(images=image, return_tensors="pt").to("cuda", torch.float16)
        caption = self.model.generate(**inputs)
        caption = self.processor.decode(caption[0], skip_special_tokens=True)
        return caption
    
# radiance hints generation
@dataclass
class Args:
    img: str  # Path to the image, to generate hints for.
    seed: int = 3407  # Seed for the generation
    fov: Optional[float] = (
        None  # Field of view for the mesh reconstruction, none for auto estimation from the image
    )

    use_sam: bool = True  # Use SAM for background removal
    mask_threshold: float = 25.0  # Mask threshold for foreground object extraction

    power: float = 1200.0  # Power of the point light
    use_gpu_for_rendering: bool = True  # Use GPU for radiance hints rendering

    pl_x: float = 1.0  # X position of the point light
    pl_y: float = 1.0  # Y position of the point light
    pl_z: float = 1.0  # Z position of the point light

    mask_path: Optional[str] = None  # Path to the mask for the image
    env_map_path: Optional[str] = None  # Path to the environment map


# elem function
def _generate_hint(
    img,
    seed=3407,
    fov=None,
    mask_path=None,
    use_sam=True,
    mask_threshold=25.0,
    power=1200.0,
    use_gpu_for_rendering=True,
    pl_x=1.0,
    pl_y=1.0,
    pl_z=1.0,
    output_dir="radiance_hints",
):
    args = Args(
        img=img,
        seed=seed,
        fov=fov,
        mask_path=mask_path,
        use_sam=use_sam,
        mask_threshold=mask_threshold,
        power=power,
        use_gpu_for_rendering=use_gpu_for_rendering,
        pl_x=pl_x,
        pl_y=pl_y,
        pl_z=pl_z,
    )

    # mask 관련 implementations
    # Load input image and generate/load mask
    input_image = imageio.v3.imread(args.img)
    input_image = cv2.resize(input_image, (512, 512))

    assert args.mask_path is not None, "mask_path should be given"
    if args.mask_path:
        # 이건 explicit하게 주면 될듯 하다.
        mask = imageio.v3.imread(args.mask_path)
        if mask.ndim == 3:
            mask = mask[..., -1]
        mask = cv2.resize(mask, (512, 512))
    else:
        _, mask = rm_bg(input_image, use_sam=args.use_sam)
    mask = mask[..., None].repeat(3, axis=-1)

    # Render radiance hints
    pls = [(args.pl_x, args.pl_y, args.pl_z)]

    # cache middle results
    # TODO: lighting condition이 env map에의해서 explicit하게 주어져야 할텐데 약간 걱정되네
    img_id = os.path.basename(args.img).split(".")[0]
    lighting_id = f"pl-{args.pl_x}-{args.pl_y}-{args.pl_z}-{args.power}"
    output_folder = os.path.join(output_dir, img_id, lighting_id)
    os.makedirs(output_folder, exist_ok=True)
    # check if the radiance hints are already rendered and full

    print(f"Rendering radiance hints")
    # Mesh reconstruction and fov estimation for hints rendering
    fov = args.fov
    # TODO: explicit하게 mesh를 주면 좋을 것이다. 결과적으로 우리가 할 것은 PSNR을 높히는 것이고, 사용하면 안되는 것은 오직 eval image pairs이다.
    mesh, fov = mesh_reconstruction(input_image, mask, False, fov, args.mask_threshold)
    print(f"Mesh reconstructed with fov: {fov}")
    
    # render hints
    # explicit하게 env_map을 주어야 한다. 이미 있다..!
    render_hint_images(
        mesh,
        fov,
        pls,
        env_map=args.env_map_path,
        output_folder=output_folder,
        use_gpu=args.use_gpu_for_rendering,
    )
    print(f"Radiance hints rendered to {output_folder}")

def elem_generate_hint(args: dict):
    """
    args.image_path
    args.mask_path
    args.viewpoint_id
    args.lighting_condition_id
    args.image_id  # key
    args.output_dir  # processed root dir
    args.fov = None  # 그러면 mesh_reconstruction에서 계산하게 된다.
    args.mask_threshold: float = 0.25  #    
    args.env_map  # path to hdf
    args.pls = [[0,0,0]]  # euler angle로 environmental map을 회전하는 것이다.
    args.use_gpu_for_rendering = True  # 무조건
    args.resolution = 128  # resolution of the image
    """
    
    args = EasyDict(args)  # dict to EasyDict
        
    image_path = args.image_path
    mask_path = args.mask_path
    image_id = args.image_id
    viewpoint_id = args.viewpoint_id
    lighting_condition_id = args.lighting_condition_id

    # Load the image and mask
    image = imageio.imread(image_path)
    mask = imageio.imread(mask_path)

    # env map
    # env_map = get_envmap(viewpoint_id, lighting_condition_id)
    
    # Create a mesh from the image and mask

    output_folder = os.path.join(args.output_dir, image_id, viewpoint_id, lighting_condition_id)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if args.extended:
        extended_mat_list = [0.05, 0.1000, 0.1500, 0.2000, 0.2500, 0.3000, 0.3500, 0.4000, 0.4500, 0.5000, 0.5500, 0.6000, 0.6500]
        render_target = []
        for each in extended_mat_list:
            render_target.append(os.path.join(output_folder, f"hint00_ggx{each}.png"))
    else:
        render_target = [
            os.path.join(output_folder, f"hint00_diffuse.png"),
            os.path.join(output_folder, f"hint00_ggx0.05.png"),
            os.path.join(output_folder, f"hint00_ggx0.13.png"),
            os.path.join(output_folder, f"hint00_ggx0.34.png"),
        ]
        
    sentinel = True
    for each_render_target in render_target:
        if os.path.exists(each_render_target):
            sentinel *= True
        else:
            sentinel *= False
            
    if sentinel:
        print(f"Radiance hints already rendered to {output_folder}")
        return
    else:
        print(
        f"Rendering radiance hints for {image_path} with viewpoint {viewpoint_id} and lighting condition {lighting_condition_id}"
    )
    # Mesh reconstruction and fov estimation for hints rendering
    
    
    fov = args.fov
    mesh, fov = mesh_reconstruction(image, mask, False, fov, args.mask_threshold)
    # TODO make mesh from colmap maybe better?
    print(f"Mesh reconstructed with fov: {fov}")
    render_hint_images(
        mesh,
        fov,
        env_map=args.env_map,
        pls=args.pls,
        output_folder=output_folder,
        resolution=args.resolution,
        use_gpu=args.use_gpu_for_rendering,
        extended=args.extended,
    )
    print(f"Radiance hints rendered to {output_folder}")

# wrapper
def _generate_hints(json_path: str, output_dir: str, gpus=["0"]):
    """
    1. load json file
    2. split the (image, mask) pairs into chunks to distribute to GPUs
    3. save the chunk to a json file.
    4. for each gpu, launch a process to generate hints for the chunk

    How the input json file looks like:
    ```
    {"image_path": "path/to/image", "mask_path": "path/to/mask"}
    {"image_path": "path/to/image", "mask_path": "path/to/mask"}
    ...
    ```

    How the temporary jsonl file looks like:
    [
    {"image_path": "path/to/image", "mask_path": "path/to/mask", viewpoint_id: "NA6", lighting_condition_id: '001'}
    {"image_path": "path/to/image", "mask_path": "path/to/mask", viewpoint_id: "NA6", lighting_condition_id: '001'}
    ...
    ]

    As a result of running this function, the hints will be saved to the output directory.
    The output directory will have the following structure:
    ```
    output_dir
    ├── chunk_0.jsonl
    ├── chunk_1.jsonl
    ├── chunk_2.jsonl
    ...
    ├── chunk_N.jsonl

    ├── img_id/
    |   ├── radiance_hint_0.png
    |   ├── radiance_hint_1.png
    |   ├── radiance_hint_2.png
    |   ├── radiance_hint_3.png
    ├── img_id/
    |   ├── radiance_hint_0.png
    |   ├── radiance_hint_1.png
    |   ├── radiance_hint_2.png
    |   ├── radiance_hint_3.png
    ...

    ```

    And this function also generates a jsonl file that contains the path to the images and the hints.
    The jsonl file will have the following format:
    ```
    {"image_id": "img_id", "object_id": "object_id", "image_path": "path/to/image",
      "mask_path": "path/to/mask", "radiance_hints_dir": "path/to/radiance_hints"},
    {"image_id": "img_id", "object_id": "object_id", "image_path": "path/to/image",
      "mask_path": "path/to/mask", "radiance_hints_dir": "path/to/radiance_hints"},
    ...
    ```

    Args:
                                    json_path: path to the json file containing the (image, mask) pairs
                                    output_dir: path to the output directory
                                    gpus: list of gpu ids to use for generating hints. e.g. ['0', '1', '2', '3']
    """
    with open(json_path) as f:
        data = f.readlines()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # split the data into chunks
    chunk_size = len(data) // len(gpus)
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    assert len(chunks) == len(gpus)

    # save the chunks to jsonl files
    # 이걸 살려야함
    image_table = []
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(output_dir, f"chunk_{i}.jsonl")
        with open(chunk_path, "w") as f:
            chunk_json_dicts = []
            for line in chunk:
                image_path = json.loads(line)["image_path"]
                mask_path = json.loads(line)["mask_path"]
                viewpoint_id = image_path.split("/")[-1].split("_")[-1].split(".")[0]
                lighting_condition_id = image_path.split("/")[-1].split("_")[-2]
                image_id = image_path.split("/")[-1].split(".")[0]
                object_id = image_path.split("/")[-2]
                image_dict = {
                    "image_id": image_id,
                    "object_id": object_id,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "viewpoint_id": viewpoint_id,
                    "lighting_condition_id": lighting_condition_id,
                    "radiance_hints_dir": os.path.join(output_dir, image_id),
                }
                chunk_json_dicts.append(image_dict)
                image_table.append(image_dict)
            json.dump(chunk_json_dicts, f)

    import subprocess

    # 여기는 더 parallel하게 만들어야 함
    processes = []
    # generate hints for each chunk, parallelly
    for i, chunk_path in enumerate(chunks):
        cmd = [
            "python",
            "generate_hint.py",
            "--json_path",
            chunk_path,
            "--output_dir",
            output_dir,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpus[i])  # 각 GPU를 설정
        process = subprocess.Popen(cmd, env=env)
        processes.append(process)

    # wait for all processes to finish
    for process in processes:
        process.wait()

    # make jsonl file for metadata, using the image_table
    # 이거 살려야 함
    with open(os.path.join(output_dir, "train_data_metadata.jsonl"), "w") as f:
        for image_dict in image_table:
            json.dump(image_dict, f)
            f.write("\n")
    # TODO: Need test for this function.

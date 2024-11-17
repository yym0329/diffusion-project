import os
from dataclasses import dataclass
from typing import Optional
import json

import imageio
import numpy as np
import cv2
import simple_parsing


# TODO: change to the correct pl points
def get_envmap(viewpoint_id: str, light_condition_id: str) -> str:
    if viewpoint_id not in [
        "NA3",
        "NE7",
        "CB5",
        "CF8",
        "NA7",
        "CC7",
        "CA2",
        "NE1",
        "NC3",
        "CE2",
    ]:
        raise ValueError(f"Invalid viewpoint_id: {viewpoint_id}")
    if light_condition_id not in [
        "001",
        "002",
        "003",
        "004",
        "005",
        "006",
        "007",
        "008",
        "009",
        "010",
        "011",
        "012",
        "013",
    ]:
        raise ValueError(f"Invalid light_condition_id: {light_condition_id}")

    return f"envmaps/{viewpoint_id}_{light_condition_id}.hdr"


@dataclass
class Args:
    json_path: str
    seed: int = 3407  # Seed for the generation
    fov: Optional[float] = (
        None  # Field of view for the mesh reconstruction, none for auto estimation from the image
    )

    env_map_path: Optional[str] = None
    power: float = 1200.0  # Power of the point light
    use_gpu_for_rendering: bool = True  # Use GPU for radiance hints rendering


def main(args: Args):
    with open(args.json_path, "r") as f:
        metadata = json.load(f)

    """
	1. load jsonl (done)
	2. for each image, load the image and mask
	3. create a mesh from the image and mask
	4. render the radiance hints. Use the lightcondition as the key for the correct pl point.
	
	How the jsonl file looks like:
	[
	{
		"image_path": "center_cropped_2/images/obj_28_metal_bucket/obj_28_metal_bucket_006_NA7.png",
		"mask_path": "center_cropped_2/masks/obj_28_metal_bucket/obj_28_metal_bucket_NA7.png",
		"viewpoint_id": "NA7",
		"lighting_condition_id": "006"
	},
	...
	]
    
    Then this function saves the rendered radiance hints to the output_dir/image_id/
	"""
    from DiLightNet.demo.mesh_recon import mesh_reconstruction
    from DiLightNet.demo.render_hints import render_hint_images

    for item in metadata:
        image_path = item["image_path"]
        mask_path = item["mask_path"]
        viewpoint_id = item["viewpoint_id"]
        lighting_condition_id = item["lighting_condition_id"]
        image_id = item["image_id"]

        # Load the image and mask
        image = imageio.imread(image_path)
        mask = imageio.imread(mask_path)

        # env map
        env_map = get_envmap(viewpoint_id, lighting_condition_id)
        # Create a mesh from the image and mask

        output_folder = os.path.join(args.output_dir, image_id)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print(
            f"Rendering radiance hints for {image_path} with viewpoint {viewpoint_id} and lighting condition {lighting_condition_id}"
        )
        # Mesh reconstruction and fov estimation for hints rendering
        fov = args.fov
        mesh, fov = mesh_reconstruction(image, mask, False, fov, args.mask_threshold)
        print(f"Mesh reconstructed with fov: {fov}")
        render_hint_images(
            mesh,
            fov,
            env_map=env_map,
            output_folder=output_folder,
            use_gpu=args.use_gpu_for_rendering,
        )
        print(f"Radiance hints rendered to {output_folder}")

        # Need Test

import os
from dataclasses import dataclass
from typing import Optional

import imageio
import numpy as np
import cv2
import simple_parsing


@dataclass
class Args:
    img: str  # Path to the image, to generate hints for.
    seed: int = 3407  # Seed for the generation
    fov: Optional[float] = None  # Field of view for the mesh reconstruction, none for auto estimation from the image

    mask_path: Optional[str] = None  # Path to the mask for the image
    use_sam: bool = True  # Use SAM for background removal
    mask_threshold: float = 25.  # Mask threshold for foreground object extraction
    
    power: float = 1200.  # Power of the point light
    use_gpu_for_rendering: bool = True  # Use GPU for radiance hints rendering

    pl_x: float = 1.  # X position of the point light
    pl_y: float = 1.  # Y position of the point light
    pl_z: float = 1.  # Z position of the point light
    


if __name__ == '__main__':
    args = simple_parsing.parse(Args)
    from DiLightNet.demo.mesh_recon import mesh_reconstruction
    from DiLightNet.demo.relighting_gen import relighting_gen
    from DiLightNet.demo.render_hints import render_hint_images, render_bg_images
    from DiLightNet.demo.rm_bg import rm_bg

    # Load input image and generate/load mask
    input_image = imageio.v3.imread(args.img)
    input_image = cv2.resize(input_image, (512, 512))
    if args.mask_path:
        mask = imageio.v3.imread(args.mask_path)
        if mask.ndim == 3:
            mask = mask[..., -1]
        mask = cv2.resize(mask, (512, 512))
    else:
        _, mask = rm_bg(input_image, use_sam=args.use_sam)
    mask = mask[..., None].repeat(3, axis=-1)
    print(mask.shape)
    
    # Render radiance hints
    pls = [(
        args.pl_x,
        args.pl_y,
        args.pl_z
    ) ]

    # cache middle results
    img_id = os.path.basename(args.img).split(".")[0]
    lighting_id = f"pl-{args.pl_x}-{args.pl_y}-{args.pl_z}-{args.power}"
    output_folder = f'radiance_hints/{img_id}/{lighting_id}'
    os.makedirs(output_folder, exist_ok=True)
    # check if the radiance hints are already rendered and full
    # render_env_bg = True
    print(f"Rendering radiance hints")
    # Mesh reconstruction and fov estimation for hints rendering
    fov = args.fov
    mesh, fov = mesh_reconstruction(input_image, mask, False, fov, args.mask_threshold)
    print(f"Mesh reconstructed with fov: {fov}")
    render_hint_images(mesh, fov, pls, args.power, output_folder=output_folder, use_gpu=args.use_gpu_for_rendering)
    print(f"Radiance hints rendered to {output_folder}")
    
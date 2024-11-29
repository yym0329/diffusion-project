# make a json file for each image. like this.

# {"obj_14_red_bucket_src_011_tgt_008_CE2": {
#         "src_light": "011",
#         "src_img_path": "obj_14_red_bucket/Lights/011/raw_undistorted/CE2.JPG",
#         "tgt_light": "008",
#         "tgt_img_path": "obj_14_red_bucket/Lights/008/raw_undistorted/CE2.JPG",
#         "mask_path": "obj_14_red_bucket/output/obj_masks/CE2.png"
#     }, ...
# }

import os, json

def main():
    img_dir = "data/lighting_patterns"
    objects = os.listdir(img_dir)
    print(objects)
    
    viewpoints = ["NA3", "NE7", "CB5", "CF8", "NA7", "CC7", "CA2", "NE1", "NC3", "CE2"]
    images = []
    masks = []
    dicts = {}
    for obj in objects:
        object_dir = os.path.join(img_dir,obj)
        lightings_conditions = os.listdir(object_dir + "/Lights")
        for lighting in lightings_conditions:
            lighting_dir = os.path.join(object_dir, 'Lights', lighting)
            for viewpoint in viewpoints:
                image_path = os.path.join(lighting_dir, "raw_undistorted", f"{viewpoint}.JPG")
                mask_path = os.path.join(object_dir, "output", "obj_masks", f"{viewpoint}.png")
                
                output_img_name = f"{obj}_{lighting}_{viewpoint}"
                dicts[output_img_name] = {
                    "tgt_img_path": image_path,
                    "mask_path": mask_path
                }
    
    with open("metadata.json", "w") as f:
        json.dump(dicts, f, indent=4)
    
    print(f"Dumped {len(dicts)} images to metadata.json")
        

if __name__ == '__main__':
    main()
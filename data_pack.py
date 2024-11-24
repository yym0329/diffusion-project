from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
import os


@dataclass
class DataSetDefinition:
    """
    string에 해당하는 정보를 그냥 holding하게 만들어 버려야 겠다.

    light_condition_path_list, openillumnination 저자 제공 env map을 사용 하는 것으로 한다.

    """
    
    processed_dir_suffix: Optional[str] = None 
    resolution_4x: bool = False
    if resolution_4x:
        resolution = 512
    else:
        resolution = 128
        
    processed_data_root_dir: str = "./data/processed"
    raw_data_root_dir: str = field(default="./data/lighting_patterns")
    split: List[str] = field(default_factory=lambda: ["train", "eval"])

    # object classes, available view points
    data_definition_json_path: str = field(default="./data/data.json")

    class_name_list: List[str] = field(default_factory=lambda: [])
    view_point_list: List[str] = field(
        default_factory=lambda: [
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
        ]
    )
    light_condition_list: List[str] = field(
        default_factory=lambda: [
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
        ]
    )

    light_condition_path_list: Dict[str, str] = field(
        default_factory=lambda: {
            "001": "./data/generate_light_gt_sg/hdrs/001.hdr",
            "002": "./data/generate_light_gt_sg/hdrs/002.hdr",
            "003": "./data/generate_light_gt_sg/hdrs/003.hdr",
            "004": "./data/generate_light_gt_sg/hdrs/004.hdr",
            "005": "./data/generate_light_gt_sg/hdrs/005.hdr",
            "006": "./data/generate_light_gt_sg/hdrs/006.hdr",
            "007": "./data/generate_light_gt_sg/hdrs/007.hdr",
            "008": "./data/generate_light_gt_sg/hdrs/008.hdr",
            "009": "./data/generate_light_gt_sg/hdrs/009.hdr",
            "010": "./data/generate_light_gt_sg/hdrs/010.hdr",
            "011": "./data/generate_light_gt_sg/hdrs/011.hdr",
            "012": "./data/generate_light_gt_sg/hdrs/012.hdr",
            "013": "./data/generate_light_gt_sg/hdrs/013.hdr",
        }
    )

    # eval split
    eval_split_json_path: str = field(default="./data/eval.json")
    eval_split: List[str] = field(default_factory=lambda: [])

    def __post_init__(self):
        
        with open(self.data_definition_json_path, "r") as f:
            self.raw_data_definition = json.load(f)

        # class definition maker
        _obj_list = self.raw_data_definition["obj_list"]
        for obj in _obj_list:
            self.class_name_list.append(obj["data_name"])
        assert len(self.class_name_list) == 64, f"len(self.class_names) != 64"

        # eval split maker
        _eval_split = json.load(open(self.eval_split_json_path, "r"))
        for k, v in _eval_split.items():
            self.eval_split.append(v["src_img_path"])
            self.eval_split.append(v["tgt_img_path"])

        split = self.step1_split_generator()
        self.step1_datadict_train_list = split["train"]
        self.step1_datadict_eval_list = split["eval"]
    
    def get_step2_json(self):
        split = self.step2_dict_generator()
        return split

    @classmethod
    def get_image_path(
        cls,
        data_root_dir: str,
        class_name: str,
        light_condition: str,
        view_point: str,
        parsed: bool = False,
    ) -> str:
        """
        ./data/lighting_patterns/{each_class_name}/Lights/{each_light}/raw_undistorted/{view_point}.jpg
        """
        if parsed:
            img_path = f"{class_name}/Lights/{light_condition}/raw_undistorted/{view_point}.JPG"
        else:
            img_path = os.path.join(
                data_root_dir,
                class_name,
                "Lights",
                light_condition,
                "raw_undistorted",
                f"{view_point}.JPG",
            )
        return img_path

    @classmethod
    def get_mask_path(
        cls, data_root_dir: str, class_name: str, view_point: str, parsed: bool = False
    ) -> str:
        pass
        """
        ./data/lighting_patterns/{each_class_name}/output/obj_masks/{view_point}.png
        """
        mask_path = os.path.join(
            data_root_dir, class_name, "output", "obj_masks", f"{view_point}.png"
        )
        return mask_path
        # TODO 여기는 preprocessing 시에 배제해야 한다.
        # Nope. 이것도 해야 한다.

    def step1_split_generator(self):
        """
        cropping and train test split
        
        """
        step1_datadict_train_list = []
        step1_datadict_eval_list = []
        for each_class_name in self.class_name_list:
            for each_view_point in self.view_point_list:
                for each_light_condition in self.light_condition_list:

                    image_path = DataSetDefinition.get_image_path(
                        data_root_dir=self.raw_data_root_dir,
                        class_name=each_class_name,
                        light_condition=each_light_condition,
                        view_point=each_view_point,
                    )
                    mask_path = DataSetDefinition.get_mask_path(
                        data_root_dir=self.raw_data_root_dir,
                        class_name=each_class_name,
                        view_point=each_view_point,
                    )

                    parsed_name = DataSetDefinition.get_image_path(
                        data_root_dir=self.raw_data_root_dir,
                        class_name=each_class_name,
                        light_condition=each_light_condition,
                        view_point=each_view_point,
                        parsed=True,
                    )

                    if parsed_name in self.eval_split:
                        step1_datadict_eval_list.append(
                            {
                                "key": f"{each_class_name}_{each_light_condition}_{each_view_point}",
                                "image_path": image_path,
                                "mask_path": mask_path,
                            }
                        )
                    else:
                        step1_datadict_train_list.append(
                            {
                                "key": f"{each_class_name}_{each_light_condition}_{each_view_point}",
                                "image_path": image_path,
                                "mask_path": mask_path,
                            }
                        )

        return {"train": step1_datadict_train_list, "eval": step1_datadict_eval_list}

    def step2_dict_generator(self, resolution_4x: bool = False):
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
        
        """
        
        step2_datadict_train_list = []
        step2_datadict_eval_list = []
        
        datadict_list ={
            "train": [],
            "eval": []
        }
        
        # /data1/common_datasets/openillumination/processed/train/images/obj_01_car/
        # /data1/common_datasets/openillumination/processed/train/masks/obj_01_car/obj_01_car_CA2.png
        
        # step 3에서 viewpoint별로 정렬 해주고, 013에 대해서 가장 object 식별하기 좋아서 이걸로 해줘야 겠다.
        
        for each_split in self.split:
            for each_class in self.class_name_list:
                
                step1_root_dir = step2_root_dir = self.processed_data_root_dir
                
                if self.processed_dir_suffix is not None:
                    step2_root_dir = os.path.join(step2_root_dir, self.processed_dir_suffix)
                
                if resolution_4x:
                    step1_root_dir = os.path.join(step1_root_dir, "4x")
                    step2_root_dir = os.path.join(step2_root_dir, "4x")

                class_dir = os.path.join(step1_root_dir, each_split, "images", each_class)
                path_list = os.listdir(class_dir)
                for each_image_path in path_list:
                    base_name = os.path.basename(each_image_path)
                    base_name = os.path.splitext(base_name)[0]

                    view_point = base_name.split("_")[-1]
                    light_condition = base_name.split("_")[-2]
                    image_id = base_name.split("_")[0] + "_" + base_name.split("_")[1] + "_" + base_name.split("_")[2]
                    
                    mask_path = os.path.join(
                        step1_root_dir, each_split, "masks", each_class, 
                        f"{each_class}_{view_point}.png"
                    )
                    datadict_list[each_split].append(
                        {
                            "image_path": os.path.join(class_dir, each_image_path),
                            "mask_path": mask_path,
                            "viewpoint_id": view_point,
                            "lighting_condition_id": light_condition,
                            "image_id": image_id,
                            "output_dir": os.path.join(step2_root_dir, each_split, "hints"),
                            "fov": None,
                            "mask_threshold": 0.25,
                            "env_map": self.light_condition_path_list[light_condition],
                            "pls": [[0,0,0]],
                            "use_gpu_for_rendering": True,
                            "resolution": self.resolution
                        }
                    )
        return datadict_list
    
    def step3_data_mover(self):
        """
        processed
        -> processed/resolution_1x
            -> processed/resolution_1x/train
                -> processed/resolution_1x/train/obj_xx_xxx/
                    (GT reference image, mask, hint_diffuse, hint_ggx0.05, hint_ggx0.13, hint_ggx0.34)
            -> processed/resolution_1x/eval
        -> processed/resolution_4x
            -> processed/resolution_4x/train
            -> processed/resolution_4x/eval
        여기는 학습하게 편하게 dataset을 재구성 해주는 것을 목표로 한다.
        
        """
        pass

    def radiance_hint_checker(self):
        """
        보아하니 내가 제대로 generation을 못한 게 있는 거 같으니 확인 해주는 function을 만들자
        Dataset은 한 샘플 당 (GT reference image, mask, hint_diffuse, hint_ggx0.05, hint_ggx0.13, hint_ggx0.34) 이렇게 나오게 해주시면 됩니다
        """
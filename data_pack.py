from dataclasses import dataclass, field
from typing import List, Dict
import json
import os


@dataclass
class DataSetDefinition:
    """
    string에 해당하는 정보를 그냥 holding하게 만들어 버려야 겠다.

    light_condition_path_list, openillumnination 저자 제공 env map을 사용 하는 것으로 한다.

    """

    raw_data_root_dir: str = field(default="./data/lighting_patterns")
    processed_data_root_dir: str = field(default="./data/processed")
    split: List[str] = field(default_factory=lambda: ["train", "val"])

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
                                "key": f"{each_class_name}_{each_light_condition}_          {each_view_point}",
                                "image_path": image_path,
                                "mask_path": mask_path,
                            }
                        )

        return {"train": step1_datadict_train_list, "eval": step1_datadict_eval_list}

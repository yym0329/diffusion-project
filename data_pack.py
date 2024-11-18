from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class InputConditions:
    """
    string에 해당하는 정보를 그냥 holding하게 만들어 버려야 겠다.
    
    """
    view_point_list: Dict[str, List[str]] = field(default_factory=lambda: {
        "train": {},
        "val": {},
    })
    
    light_condition_list: Dict[str, List[str]] = field(default_factory=lambda: {
        "train": {},
        "val": {},
    })
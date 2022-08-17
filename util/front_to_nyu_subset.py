from pathlib import Path

import numpy as np
from PIL import Image

selected_nyu_bg_classes = [1, 2, 3, 12, 15, 22]
selected_nyu_fg_classes = [4, 5, 6, 7, 14, 24, 25, 33, 35]
selected_classes = sorted(selected_nyu_bg_classes + selected_nyu_fg_classes)

nyu_labels = {
    1: "Wall",
    2: "Floor",
    3: "Cabinet",
    4: "Bed",
    5: "Chair",
    6: "Sofa",
    7: "Table",
    8: "Door",
    9: "Window",
    10: "BookShelf",
    11: "Picture",
    12: "Counter",
    13: "Blinds",
    14: "Desks",
    15: "Shelves",
    16: "Curtain",
    17: "Dresser",
    18: "Pillow",
    19: "Mirror",
    20: "Floor-mat",
    21: "Clothes",
    22: "Ceiling",
    23: "Books",
    24: "Refrigerator",
    25: "Television",
    26: "Paper",
    27: "Towel",
    28: "Shower-curtain",
    29: "Box",
    30: "Whiteboard",
    31: "Person",
    32: "NightStand",
    33: "Toilet",
    34: "Sink",
    35: "Lamp",
    36: "Bathtub",
    37: "Bag",
    38: "Other-structure",
    39: "Other-furniture",
    40: "Other-prop",
}


def front_to_nyu(semantics):
    frontname_to_frontlabel = {x.split(',')[1]: int(x.split(',')[0]) for x in Path("resources/3D_front_mapping.csv").read_text().splitlines()[1:]}
    frontname_to_nyulabel = {x.split(',')[1]: int(x.split(',')[0]) for x in Path("resources/3D_front_nyu_mapping.csv").read_text().splitlines()[1:]}
    frontname_to_nyulabel['void'] = 40
    frontlabel_to_nyulabel = np.zeros(128, dtype=np.int32)
    for frontname, frontlabel in frontname_to_frontlabel.items():
        if frontname in frontname_to_nyulabel:
            frontlabel_to_nyulabel[frontlabel] = frontname_to_nyulabel[frontname]
        else:
            frontlabel_to_nyulabel[frontlabel] = 40
    nyu_semantics = semantics.reshape(-1)
    valid_nyu_semantics = nyu_semantics != 255
    nyu_semantics[valid_nyu_semantics] = frontlabel_to_nyulabel[nyu_semantics[valid_nyu_semantics].tolist()]
    return nyu_semantics.reshape(semantics.shape)


def read_front_semantics(sem_path, mask, remove_masked=True):
    semantics = np.array(Image.open(sem_path))
    if remove_masked:
        semantics = np.where(mask, semantics, np.ones_like(semantics) * 255)
    return front_to_nyu(semantics)

import torch
from collections import namedtuple
from cityscapesscripts.helpers.labels import labels as cs_labels
from cityscapesscripts.helpers.labels import Label

synthia_cs_labels = [
    # name                         id trainId   category  catId  hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 255, 'nature', 4, False, True, (152, 251, 152)),
    # Removed because not present in Synthia dataset
    Label('sky', 23, 9, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 10, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 11, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 12, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 255, 'vehicle', 7, True, True, (0, 0, 70)),  # Removed because not present in Synthia dataset
    Label('bus', 28, 13, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 255, 'vehicle', 7, True, True, (0, 80, 100)),  # Removed because not present in Synthia dataset
    Label('motorcycle', 32, 14, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 15, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

synthia_bdd_labels = [
    Label('unlabeled', 255, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 255, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ego vehicle', 255, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ground', 255, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('static', 255, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('parking', 255, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 255, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('road', 0, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 1, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('bridge', 255, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('building', 2, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 3, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 4, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('garage', 255, 255, 'construction', 2, False, True, (180, 100, 180)),
    Label('guard rail', 255, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('tunnel', 255, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('banner', 255, 255, 'object', 3, False, True, (250, 170, 100)),
    Label('billboard', 255, 255, 'object', 3, False, True, (220, 220, 250)),
    Label('lane divider', 255, 255, 'object', 3, False, True, (255, 165, 0)),
    Label('parking sign', 255, 255, 'object', 3, False, False, (220, 20, 60)),
    Label('pole', 5, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 255, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('street light', 255, 255, 'object', 3, False, True, (220, 220, 100)),
    Label('traffic cone', 255, 255, 'object', 3, False, True, (255, 70, 0)),
    Label('traffic device', 255, 255, 'object', 3, False, True, (220, 220, 220)),
    Label('traffic light', 6, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 7, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('traffic sign frame', 255, 255, 'object', 3, False, True, (250, 170, 250)),
    Label('vegetation', 8, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 9, 255, 'nature', 4, False, True, (152, 251, 152)),  # Removed from dataset
    Label('sky', 10, 9, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 11, 10, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 12, 11, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 13, 12, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('bus', 15, 13, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('motorcycle', 17, 14, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 18, 15, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('caravan', 255, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 255, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('truck', 14, 255, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('train', 16, 255, 'vehicle', 7, True, False, (0, 80, 100)),
]

SynthiaClass = namedtuple(
    "SynthiaClass",
    ["name", "id", "trainId", "ignoreInEval", "color"]
)

synthia_labels = [
        SynthiaClass("road",            3, 0,   False,  (128, 64, 128)),
        SynthiaClass("sidewalk",        4, 1,   False,  (244, 35, 232)),
        SynthiaClass("building",        2, 2,   False,  (70, 70, 70)),
        SynthiaClass("wall",            21, 3,  False,  (102, 102, 156)),
        SynthiaClass("fence",           5, 4,   False,  (64, 64, 128)),
        SynthiaClass("pole",            7, 5,   False,  (153, 153, 153)),
        SynthiaClass("traffic light",   15, 6,  False,  (250, 170, 30)),
        SynthiaClass("traffic sign",    9, 7,   False,  (220, 220, 0)),
        SynthiaClass("vegetation",      6, 8,   False,  (107, 142, 35)),
        SynthiaClass("terrain",         16, 255,  True,  (152, 251, 152)),
        SynthiaClass("sky",             1, 9,  False,  (70, 130, 180)),
        SynthiaClass("pedestrian",      10, 10, False,  (220, 20, 60)),
        SynthiaClass("rider",           17, 11, False,  (255, 0, 0)),
        SynthiaClass("car",             8, 12,  False,  (0, 0, 142)),
        SynthiaClass("truck",           18, 255, True,  (0, 0, 70)),
        SynthiaClass("bus",             19, 13, False,  (0, 60, 100)),
        SynthiaClass("train",           20, 255, True,  (0, 80, 100)),
        SynthiaClass("motorcycle",      12, 14, False,  (0, 0, 230)),
        SynthiaClass("bicycle",         11, 15, False,  (119, 11, 32)),
        SynthiaClass("void",            0, 255, True,   (0, 0, 0)),
        SynthiaClass("parking slot",    13, 255, True,  (250, 170, 160)),
        SynthiaClass("road-work",       14, 255, True,  (128, 64, 64)),
        SynthiaClass("lanemarking",     22, 255, True,  (102, 102, 156))

    ]

MapillaryClass = namedtuple(
    "MapillaryClass",
    ["id", "trainId"]
)

mapillary_labels = [
        MapillaryClass(13, 0),
        MapillaryClass(24, 0),
        MapillaryClass(41, 0),
        MapillaryClass(2, 1),
        MapillaryClass(15, 1),
        MapillaryClass(17, 2),
        MapillaryClass(6, 3),
        MapillaryClass(3, 4),
        MapillaryClass(45, 5),
        MapillaryClass(47, 5),
        MapillaryClass(48, 6),
        MapillaryClass(50, 7),
        MapillaryClass(30, 8),
        MapillaryClass(29, 9),
        MapillaryClass(27, 10),
        MapillaryClass(19, 11),
        MapillaryClass(20, 12),
        MapillaryClass(21, 12),
        MapillaryClass(22, 12),
        MapillaryClass(55, 13),
        MapillaryClass(61, 14),
        MapillaryClass(54, 15),
        MapillaryClass(58, 16),
        MapillaryClass(57, 17),
        MapillaryClass(52, 18),
    ]

mapillary_synthia_labels = [
        MapillaryClass(13, 0),
        MapillaryClass(24, 0),
        MapillaryClass(41, 0),
        MapillaryClass(2, 1),
        MapillaryClass(15, 1),
        MapillaryClass(17, 2),
        MapillaryClass(6, 3),
        MapillaryClass(3, 4),
        MapillaryClass(45, 5),
        MapillaryClass(47, 5),
        MapillaryClass(48, 6),
        MapillaryClass(50, 7),
        MapillaryClass(30, 8),
        MapillaryClass(29, 255), #terrain
        MapillaryClass(27, 9),
        MapillaryClass(19, 10),
        MapillaryClass(20, 11),
        MapillaryClass(21, 11),
        MapillaryClass(22, 11),
        MapillaryClass(55, 12),
        MapillaryClass(61, 255), #truck
        MapillaryClass(54, 13),
        MapillaryClass(58, 255), #train
        MapillaryClass(57, 14),
        MapillaryClass(52, 15),
    ]

WilddashClass = namedtuple(
    "WilddashClass",
    ["id", "trainId"]
)

wilddash_labels = [
    WilddashClass(0, 255),
    WilddashClass(1, 255),
    WilddashClass(2, 255),
    WilddashClass(3, 255),
    WilddashClass(4, 255),
    WilddashClass(5, 255),
    WilddashClass(6, 255),
    WilddashClass(7, 0),
    WilddashClass(8, 1),
    WilddashClass(9, 255),
    WilddashClass(10, 255),
    WilddashClass(11, 2),
    WilddashClass(12, 3),
    WilddashClass(13, 4),
    WilddashClass(14, 255),
    WilddashClass(15, 255),
    WilddashClass(16, 255),
    WilddashClass(17, 5),
    WilddashClass(18, 255),
    WilddashClass(19, 6),
    WilddashClass(20, 7),
    WilddashClass(21, 8),
    WilddashClass(22, 9),
    WilddashClass(23, 10),
    WilddashClass(24, 11),
    WilddashClass(25, 12),
    WilddashClass(26, 13),
    WilddashClass(27, 14),
    WilddashClass(28, 15),
    WilddashClass(29, 255),
    WilddashClass(30, 255),
    WilddashClass(31, 16),
    WilddashClass(32, 17),
    WilddashClass(33, 18),
    WilddashClass(34, 13),
    WilddashClass(35, 13),
    WilddashClass(36, 255),
    WilddashClass(37, 255),
    WilddashClass(38, 0),
]

wilddash_synthia_labels = [
    WilddashClass(0, 255),
    WilddashClass(1, 255),
    WilddashClass(2, 255),
    WilddashClass(3, 255),
    WilddashClass(4, 255),
    WilddashClass(5, 255),
    WilddashClass(6, 255),
    WilddashClass(7, 0),
    WilddashClass(8, 1),
    WilddashClass(9, 255),
    WilddashClass(10, 255),
    WilddashClass(11, 2),
    WilddashClass(12, 3),
    WilddashClass(13, 4),
    WilddashClass(14, 255),
    WilddashClass(15, 255),
    WilddashClass(16, 255),
    WilddashClass(17, 5),
    WilddashClass(18, 255),
    WilddashClass(19, 6),
    WilddashClass(20, 7),
    WilddashClass(21, 8),
    WilddashClass(22, 255), #terrain
    WilddashClass(23, 9),
    WilddashClass(24, 10),
    WilddashClass(25, 11),
    WilddashClass(26, 12),
    WilddashClass(27, 255), #truck
    WilddashClass(28, 13),
    WilddashClass(29, 255),
    WilddashClass(30, 255),
    WilddashClass(31, 255), #train
    WilddashClass(32, 14),
    WilddashClass(33, 15),
    WilddashClass(34, 12),
    WilddashClass(35, 12),
    WilddashClass(36, 255),
    WilddashClass(37, 255),
    WilddashClass(38, 0),
]

def convert_ids_to_trainids(gt: torch.Tensor,
                            source: str,
                            target: str) -> torch.Tensor:
    """

    Args:
        gt: Ground truth tensor with labels from 0 to 34 / 0 to 33 and -1
        source: Name of source domain
        target: Name of target domain

    Returns:
        gt: Groundtruth tensor with labels from 0 to 18 and 255 for non training ids
    """

    # Check if target domain is BDD, if true check source domain
    if target == "bdd":

        # Check if source domain is GTA, if true, return gt without conversion, because BDD has train_ids -> [0, 19]
        if source == "gta":
            return gt

        else:
            # If source domain is Synthia, use Synthia/BDD lookup table
            labels = synthia_bdd_labels

    # Check if target domain is IDD, if true check source domain
    elif target == "idd":

        # Check if source domain is GTA, if true, return gt without conversion, because IDD has train_ids -> [0, 19]
        if source == "gta":
            return gt

        else:
            # If source domain is Synthia, use Synthia/IDD lookup table
            labels = synthia_bdd_labels

    # If target is not BDD, check source domain
    elif target == "synthia" and source == "synthia":
        labels = synthia_labels

    elif target == "mapillary":
        # If source domain is GTA, use standard CS lookup table
        if source == "gta":
            labels = mapillary_labels

        # If source domain is Synthia, use Synthia/CS lookup table
        else:
            labels = mapillary_synthia_labels

    elif target == "wilddash":
        # If source domain is GTA, use standard CS lookup table
        if source == "gta":
            labels = wilddash_labels

        # If source domain is Synthia, use Synthia/CS lookup table
        else:
            labels = wilddash_synthia_labels
    elif target in ["cityscapes", "gta"]:

        # If source domain is GTA, use standard CS lookup table
        if source == "gta":
            labels = cs_labels

        # If source domain is Synthia, use Synthia/CS lookup table
        else:
            labels = synthia_cs_labels

    else:
        raise ValueError(f"Target domain {target} unknown")

    gt_copy = torch.ones_like(gt) * 255
    for cs_label in labels:
        orig_id = cs_label.id
        new_id = cs_label.trainId
        # Manually set license plate to id 34 and trainId 255
        if orig_id == -1:
            orig_id = 34
            new_id = 255
        gt_copy[gt == orig_id] = new_id
    return gt_copy

def convert_trainids_to_ids(pred: torch.Tensor,
                            source: str,
                            target: str) -> torch.Tensor:
    """

    Args:
        gt: Groundtruth tensor with labels from 0 to 34 / 0 to 33 and -1
        source: Name of source domain
        target: Name of target domain

    Returns:
        gt: Groundtruth tensor with labels from 0 to 18 and 255 for non training ids
    """

    # Check if target domain is BDD, if true check source domain
    if target == "bdd":

        # Check if source domain is GTA, if true, return gt without conversion, because BDD has train_ids -> [0, 19]
        if source == "gta":
            return pred

        else:
            # If source domain is Synthia, use Synthia/BDD lookup table
            labels = synthia_bdd_labels

    # If target is not BDD, check source domain
    else:

        # If source domain is GTA, use standard CS lookup table
        if source == "gta":
            labels = cs_labels

        # If source domain is Synthia, use Synthia/CS lookup table
        else:
            labels = synthia_cs_labels

    for cs_label in labels[::-1]:
        orig_id = cs_label.id
        new_id = cs_label.trainId
        # Manually set license plate to id 34 and trainId 255
        if orig_id == -1:
            orig_id = 34
            new_id = 255
        pred[pred == orig_id] = new_id
    return pred

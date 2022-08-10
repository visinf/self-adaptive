import os
import glob
import argparse
import pathlib
import PIL.Image
import torch
from typing import List, Callable, Optional, Tuple
from tqdm import tqdm
import urllib.request
import shutil
import scipy.io

class GTADataset(object):
    """
    Download, unzip, and split data: python datasets/gta.py --dataset-root /path/to/gta --download-data --split-data
    This also removes samples with size mismatches between image and annotation
    The GTA dataset is required to have following folder structure:
    gta/
        images/
               train/*.png
               test/*.png
               val/*.png
        labels/
               train/*.png
               test/*.png
               val/*.png
    """
    def __init__(self,
                 root,
                 split,
                 transforms: Optional[Callable] = None):

        super(GTADataset, self).__init__()

        self.images_dir = os.path.join(root, "images", split)
        self.targets_dir = os.path.join(root, "labels", split)
        self.split = split
        self.images = []
        self.targets = []
        self.transforms = transforms

        for file_name in os.listdir(self.images_dir):
            target_name = file_name

            self.images.append(os.path.join(self.images_dir, file_name))
            self.targets.append(os.path.join(self.targets_dir, target_name))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        image = PIL.Image.open(self.images[index]).convert('RGB')
        target = PIL.Image.open(self.targets[index])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)

def gta(root: str,
        split: str,
        transforms: List[Callable]):
    return GTADataset(root=root,
                      split=split,
                      transforms=transforms)

def preprocess(dataset_root: str):
    """
    Function to remove data samples with size mismatches between image and annotation
    """
    # Create catalog of every GTA image in dataset directory
    dataset_split = ["train", "val", "test"]
    # Count deleted files
    count_del = 0

    for split in dataset_split:
        images = sorted(glob.glob(os.path.join(dataset_root, "images", split, "*.png")))
        labels = sorted(glob.glob(os.path.join(dataset_root, "labels", split, "*.png")))
        assert len(images) == len(labels), "Length of catalogs does not match!"

        print("Preprocessing images and labels")
        for image, label in tqdm(zip(images, labels)):

            # Assert that label corresponds to current image
            image_name = image.split("/")[-1]
            label_name = label.split("/")[-1]
            assert image_name == label_name

            # Load image and label
            img = PIL.Image.open(image)
            gt = PIL.Image.open(label)

            if img.size != gt.size:
                print(f"Found data sample pair with unmatching size. Deleting file with name: {image_name} and {label_name}.")
                # Delete mismatching data samples
                os.remove(path=image)
                os.remove(path=label)
                count_del += 1
    print(f"{count_del} images have been removed from the dataset")

def download_dataset(dataset_root: str, download_path_main: str ="https://download.visinf.tu-darmstadt.de/data/from_games"):
    download_path = os.path.join(download_path_main, "data")
    pathlib.Path(dataset_root).mkdir(exist_ok=True, parents=True)
    for i in tqdm(range(1, 11)):
        index = f"{i:02}"
        for file_name in ["images", "labels"]:
            file_name_zip = f"{index}_{file_name}.zip"
            file_path = os.path.join(download_path, file_name_zip)
            out_path = os.path.join(dataset_root, file_name_zip)
            urllib.request.urlretrieve(file_path, filename=out_path)
            shutil.unpack_archive(out_path, dataset_root)
            os.remove(out_path)
    mapping_name = "read_mapping.zip"
    download_path_map = os.path.join(download_path_main, "code", mapping_name)
    out_path = os.path.join(dataset_root, mapping_name)
    urllib.request.urlretrieve(download_path_map, filename=out_path)
    shutil.unpack_archive(out_path, os.path.join(dataset_root, "read_mapping"))
    os.remove(out_path)

def load_split(path: str):
    mat = scipy.io.loadmat(path)
    trainIds = mat['trainIds']
    valIds = mat['valIds']
    testIds = mat['testIds']
    return trainIds, valIds, testIds

def load_mapping(path: str):
    mat = scipy.io.loadmat(path)
    classes = mat['classes']
    cityscapesMap = mat['cityscapesMap']
    camvidMap = mat['camvidMap']
    return classes, cityscapesMap, camvidMap

def split_dataset(dataset_root):
    # Get trainIds, valIds, testIds
    path_to_map = os.path.join(dataset_root, "read_mapping")
    path_to_mat = os.path.join(path_to_map, "split.mat")
    trainIds, valIds, testIds = load_split(path=path_to_mat)
    split_ids = [trainIds.squeeze(), valIds.squeeze(), testIds.squeeze()]
    split_paths = ['train', 'val', 'test']

    img_dir = os.path.join(dataset_root, "images")
    label_dir = os.path.join(dataset_root, "labels")
    img_out_dir = os.path.join(dataset_root, "images")
    label_out_dir = os.path.join(dataset_root, "labels")

    for split_id, split_path in zip(split_ids, split_paths):
        path_split_image = os.path.join(img_out_dir, split_path)
        path_split_label = os.path.join(label_out_dir, split_path)
        pathlib.Path(path_split_label).mkdir(parents=True, exist_ok=True)
        pathlib.Path(path_split_image).mkdir(parents=True, exist_ok=True)
        for img_id in tqdm(split_id):
            img_name = str(img_id).zfill(5) + '.png'
            shutil.move(os.path.join(img_dir, img_name), os.path.join(path_split_image, img_name))
            shutil.move(os.path.join(label_dir, img_name), os.path.join(path_split_label, img_name))
    shutil.rmtree(path_to_map)
    if img_dir != img_out_dir:
        shutil.rmtree(img_dir)
    if label_dir != label_out_dir:
        shutil.rmtree(label_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=os.path.join(os.getcwd(), "datasets", "gta"))
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--split-data", action="store_true")
    args = parser.parse_args()
    if args.download_data:
        download_dataset(args.dataset_root)
    if args.split_data:
        split_dataset(args.dataset_root)
    preprocess(args.dataset_root)
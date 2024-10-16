from data.base_dataset import BaseDataset, Rescale_fixed, Normalize_image
from data.image_folder import make_dataset, make_dataset_test

import os
import cv2
import json
import itertools
import collections
from tqdm import tqdm

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import ast
from pycocotools.coco import COCO


class SegmentationAugmentation:
    def __init__(self, opt, flip_prob=0.5, jitter_prob=0.25, gray_prob=0.1, crop_prob=0.25, rotate_prob=0.5):
        self.opt = opt
        self.flip_prob = flip_prob
        self.jitter_prob = jitter_prob
        self.gray_prob = gray_prob
        self.crop_prob = crop_prob
        self.rotate_prob = rotate_prob

    def __call__(self, image, mask):
        # Ensure image and mask are PyTorch tensors
        if not isinstance(image, torch.Tensor):
            image = transforms.F.to_tensor(image)
        if not isinstance(mask, torch.Tensor):
            mask = transforms.F.to_tensor(mask)

        # Random horizontal flip
        if torch.rand(1) < self.flip_prob:
            image = transforms.F.hflip(image)
            mask = transforms.F.hflip(mask)

        if torch.rand(1) < self.jitter_prob:
            brightness = torch.rand(1).item() * 2
            image = transforms.F.adjust_brightness(image, brightness)

            contrast = torch.rand(1).item() * 2
            image = transforms.F.adjust_contrast(image, contrast)

            saturation = torch.rand(1).item() * 2
            image = transforms.F.adjust_saturation(image, saturation)

            hue = torch.rand(1).item() - 0.5
            image = transforms.F.adjust_hue(image, hue)

        if torch.rand(1) < self.gray_prob:
            image = transforms.F.rgb_to_grayscale(image, 3)
        
        # Random crop
        if torch.rand(1) < self.crop_prob:
            func = transforms.RandomResizedCrop([self.opt.fine_height, self.opt.fine_width])
            i, j, h, w = func.get_params(image, func.scale, func.ratio)
            image = transforms.F.resized_crop(image, i, j, h, w, [self.opt.fine_height, self.opt.fine_width])
            mask = transforms.F.resized_crop(mask, i, j, h, w, [self.opt.fine_height, self.opt.fine_width])

        # Random rotation
        if torch.rand(1) < self.rotate_prob:
            angle = torch.randint(-15, 15, (1,)).item()
            image = transforms.F.rotate(image, angle)
            mask = transforms.F.rotate(mask, angle)

        image = Normalize_image(self.opt.mean, self.opt.std)(image)

        return image, mask


class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        self.initialize(opt)
    
    def initialize(self, opt):
        self.opt = opt
        self.image_dir = opt.image_folder
        self.df_path = opt.df_path
        self.width = opt.fine_width
        self.height = opt.fine_height

        # for rgb imgs

        transforms_list = []
        transforms_list += [transforms.ToTensor()]
        transforms_list += [Normalize_image(opt.mean, opt.std)]
        self.test_transform_rgb = transforms.Compose(transforms_list)
        # self.transform_rgb = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5),
        #     transforms.RandomGrayscale(p=0.1),
        #     transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.9, 1.1), shear=15, interpolation=Image.BILINEAR),
        #     transforms.RandomRotation(degrees=30, interpolation=Image.BILINEAR),
        #     transforms.RandomChoice([
        #         transforms.GaussianBlur(kernel_size=(5,9)),
        #         transforms.RandomResizedCrop(size=(self.width, self.height), scale=(0.8, 1)),
        #         transforms.RandomPerspective(),
        #     ]),
        #     transforms.ToTensor(),
        #     transforms.RandomErasing(p=0.25),
        #     Normalize_image(opt.mean, opt.std),
        # ])
        self.transform_rgb = SegmentationAugmentation(opt)

        # self.df = pd.read_csv(self.df_path)
        self.image_info = collections.defaultdict(dict)
        # self.df["CategoryId"] = self.df.ClassId.apply(lambda x: str(x).split("_")[0]) # no effect in the 2019 dataset
        # temp_df = (
        #     self.df.groupby("ImageId")[["EncodedPixels", "CategoryId"]]
        #     .agg(lambda x: list(x))
        #     .reset_index()
        # )
        # size_df = self.df.groupby("ImageId")[["Height", "Width"]].mean().reset_index()
        # temp_df = temp_df.merge(size_df, on="ImageId", how="left") # including columns of img id, rle mask, cls id, h, w
        self.df = pd.read_csv(self.df_path, converters={'EncodedPixels': ast.literal_eval, 'CategoryId': ast.literal_eval})
        for index, row in tqdm(self.df.iterrows(), total=len(self.df)):
            image_id = row["ImageId"]
            image_path = os.path.join(self.image_dir, image_id)
            self.image_info[index]["image_id"] = image_id
            self.image_info[index]["image_path"] = image_path
            self.image_info[index]["width"] = self.width
            self.image_info[index]["height"] = self.height
            self.image_info[index]["labels"] = row["CategoryId"]
            self.image_info[index]["orig_height"] = int(row["Height"])
            self.image_info[index]["orig_width"] = int(row["Width"])
            self.image_info[index]["annotations"] = row["EncodedPixels"]

        self.dataset_size = len(self.image_info)
        self.coco = COCO("runs/labelme2coco/dataset.json")

    def __getitem__(self, index):
        ann_ids = self.coco.getAnnIds(imgIds=[index + 1], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        fname = self.coco.loadImgs(index + 1)[0]['file_name'].split('/', 1)[-1]
        img_path = os.path.join(self.image_dir, fname)
        idx = self.df.index[self.df['ImageId'] == fname].tolist()[0]

        # load images ad masks
        # idx = index
        # img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), resample=Image.BICUBIC)
        # image_tensor = self.test_transform_rgb(img) if self.opt.test else self.transform_rgb(img)

        info = self.image_info[idx]
        mask = np.zeros(
            (len(info["annotations"]), self.width, self.height), dtype=np.uint8
        )
        labels = []
        for m, (annotation, label) in enumerate(
            zip(info["annotations"], info["labels"])
        ):
            # sub_mask = self.rle_decode(
            #     annotation, (info["orig_height"], info["orig_width"])
            # )
            sub_mask = self.coco.annToMask(anns[m])
            sub_mask = Image.fromarray(sub_mask)
            sub_mask = sub_mask.resize(
                (self.width, self.height), resample=Image.BICUBIC
            )
            mask[m, :, :] = sub_mask
            labels.append(int(label))

        num_objs = len(labels)
        boxes = []
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(mask[0, :, :])

        nmx = np.zeros((len(new_masks), self.width, self.height), dtype=np.uint8)
        for i, n in enumerate(new_masks):
            nmx[i, :, :] = n

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.int64)
        masks = torch.as_tensor(nmx, dtype=torch.uint8)

        final_label = np.zeros((self.width, self.height), dtype=np.uint8)
        first_channel = np.zeros((self.width, self.height), dtype=np.uint8)
        second_channel = np.zeros((self.width, self.height), dtype=np.uint8)
        third_channel = np.zeros((self.width, self.height), dtype=np.uint8)
        fourth_channel = np.zeros((self.width, self.height), dtype=np.uint8)
        fifth_channel = np.zeros((self.width, self.height), dtype=np.uint8)

        # upperbody = [0, 1, 2, 3, 4, 5]
        # lowerbody = [6, 7, 8]
        # wholebody = [9, 10, 11, 12]
        tops = [0, 1]
        sweater = [2, 3]
        outerwear = [4, 5, 9]
        bottoms = [6, 7, 8]
        wholebody = [10, 11, 12]

        for i in range(len(labels)):
            if labels[i] in tops:
                first_channel += new_masks[i]
            elif labels[i] in sweater:
                second_channel += new_masks[i]
            elif labels[i] in outerwear:
                third_channel += new_masks[i]
            elif labels[i] in bottoms:
                fourth_channel += new_masks[i]
            elif labels[i] in wholebody:
                fifth_channel += new_masks[i]

        first_channel = (first_channel > 0).astype("uint8")
        second_channel = (second_channel > 0).astype("uint8")
        third_channel = (third_channel > 0).astype("uint8")
        fourth_channel = (fourth_channel > 0).astype("uint8")
        fifth_channel = (fifth_channel > 0).astype("uint8")

        final_label = first_channel + second_channel * 2 + third_channel * 3 + fourth_channel * 4 + fifth_channel * 5
        conflict_mask = (final_label <= 5).astype("uint8")
        final_label = (conflict_mask) * final_label + (1 - conflict_mask) * 1
        

        if self.opt.test:
            image_tensor = self.test_transform_rgb(img)
            target_tensor = torch.as_tensor(final_label, dtype=torch.int64)
        else:
            image_tensor, target_tensor = self.transform_rgb(img, torch.as_tensor(final_label, dtype=torch.int64).unsqueeze(0))
            target_tensor = target_tensor.squeeze()
        return image_tensor, target_tensor

    def __len__(self):
        return self.dataset_size

    def name(self):
        return "AlignedDataset"

    def rle_decode(self, mask_rle, shape):
        """
        mask_rle: run-length as string formated: [start0] [length0] [start1] [length1]... in 1d array
        shape: (height,width) of array to return
        Returns numpy array according to the shape, 1 - mask, 0 - background
        """
        shape = (shape[1], shape[0])
        s = mask_rle.split()
        # gets starts & lengths 1d arrays
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
        starts -= 1
        # gets ends 1d array
        ends = starts + lengths
        # creates blank mask image 1d array
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        # sets mark pixles
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        # reshape as a 2d mask image
        return img.reshape(shape).T  # Needed to align to RLE direction

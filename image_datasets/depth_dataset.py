import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import cv2
from torchvision import transforms


def depth_processor(image, low_threshold=100, high_threshold=200):
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


class DepthImageDataset(Dataset):
    def __init__(
        self,
        data_root: str = '~/work/dev6/DiffusionAsShader/data/bridge',
        image_folder: str = 'embeddings_full',
        depth_folder: str = 'renderings/points',
        split: str = 'train',
    ) -> None:

        self.data_root = data_root
        self.image_dir = os.path.join(data_root, image_folder, split, 'images1')
        self.depth_dir = os.path.join(data_root, depth_folder, split)
        self.split = split

        self.images = [
            os.path.join(self.image_dir, fn)
            for fn in os.listdir(self.image_dir) if fn.endswith('.png')
        ]
        depth_episode_ids = list(map(lambda fn: int(fn), os.listdir(self.depth_dir)))
        self.images = list(filter(lambda fn: int(os.path.basename(fn).split('_')[0]) in depth_episode_ids, self.images))
        print(f'Loaded {len(self)} samples.')

        # self.image_transforms = transforms.Compose(
        #     [
        #         transforms.Resize(320, interpolation=transforms.InterpolationMode.BILINEAR),
        #         transforms.CenterCrop(tuple([320, 480])),
        #     ]
        # )
        self.depth_transforms = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(tuple([256, 320])),
                transforms.Resize(384, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(tuple([320, 480])),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            # get rgb image
            img_path = self.images[idx]
            img = Image.open(img_path)
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            sample_name = os.path.basename(img_path)
            episode_id, start_frame_id, _ = sample_name.split('_')
            # get depth image
            episode_id = int(episode_id)
            start_frame_id = int(start_frame_id)
            hint = torch.from_numpy(np.load(
                os.path.join(self.depth_dir, str(episode_id), f'frame_{start_frame_id:04d}.npy')
            ))  # [h, w]
            hint = self.depth_transforms(hint[None, ...])  # [1, h, w]
            hint = hint.clamp(min=0., max=1.)
            hint = ((hint - hint.min()) / (hint.max() - hint.min())) * 2 - 1
            hint = hint.repeat(3, 1, 1)  # [3, h, w]
            # prompt
            prompt = ''
            return img, hint, prompt
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))


def loader(train_batch_size, num_workers, **args):
    dataset = DepthImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)

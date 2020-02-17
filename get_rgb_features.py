'''
Retrieve RGB features from pretrained VGG19 for videos
'''

import pandas as pd
import torch
import torch.nn as nn
import torchvision
import os
import argparse
import json

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm


FRAMES_PER_SEC = 5
SEC_PER_SEGMENT = 5
SELECT_FPS = 25

def read_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

class DiDeMoDataset(Dataset):

    def __init__(self, dataset_info, dataset_dir='.'):
        self.dataset_info = dataset_info
        self.dataset_dir = dataset_dir
        self.mean = torch.Tensor([0.485, 0.456, 0.406]) # ImageNet mean
        self.std = torch.Tensor([0.229, 0.224, 0.225]) # ImageNet std

    def __getitem__(self, index):
        filename = self.dataset_info[index]['video']
        num_segments = self.dataset_info[index]['num_segments']

        vframes, _, vinfo = torchvision.io.read_video(
            filename='{}/{}.mp4'.format(self.dataset_dir, filename),
            pts_unit='sec',
            end_pts=SEC_PER_SEGMENT*num_segments
        )
        # subsample frames (5 frames corresponds to 1 sec of the video)
        step = vinfo['video_fps'] / FRAMES_PER_SEC
        mask = torch.Tensor([round(i * step) for i in range(SELECT_FPS * (num_segments - 1))]).long()
        last_frame = mask[-1].item()
        step = (vframes.size(0) - last_frame - 1) / SELECT_FPS
        mask = torch.cat([
            mask,
            torch.Tensor([last_frame + round((i + 1) * step) for i in range(SELECT_FPS)]).long()
        ])
        vframes = vframes.index_select(dim=0, index=mask)

        # [T, H, W, C] -> [T, C, H, W]
        vframes = vframes.transpose(3, 1).transpose(2, 3)

        # scale to [0, 1]
        vframes = vframes.float().div(255)
        # normalize with ImageNet mean and std
        vframes = vframes.sub(self.mean[None, :, None, None]).div(self.std[None, :, None, None])

        return (vframes, filename)

    def __len__(self):
        return len(self.dataset_info)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='videos/')
    parser.add_argument("--features_dir", type=str, default='features/')
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    DATASET_DIRECTORY = args.dataset_dir
    FT_DIRECTORY = args.features_dir
    if args.device in ['cpu','cuda']:
        DEVICE = args.device
    else:
        if torch.cuda.is_available():
            DEVICE = 'cuda'
        else:
            DEVICE = 'cpu'

    video_filenames = [f[:-4] for f in os.listdir(DATASET_DIRECTORY)
                            if os.path.isfile(os.path.join(DATASET_DIRECTORY, f)) and (f[-4:] == '.mp4')]

    dataset_info = read_json('didemo_video_info.json')

    # read all videos from DiDeMo dataset
    didemo_dataset = DiDeMoDataset(dataset_info, dataset_dir=DATASET_DIRECTORY)
    dataloader = DataLoader(didemo_dataset, batch_size=1, shuffle=False, num_workers=4)

    # load pretrained VGG19 net
    model_vgg19 = torchvision.models.vgg13(pretrained=True).to(DEVICE)

    # need features from fc7 (second FC-layer in classifier)
    model_vgg19.classifier = nn.Sequential(*[model_vgg19.classifier[i] for i in range(5)])

    model_vgg19.eval()
    for (vframes, video) in tqdm(dataloader):
        features = model_vgg19(vframes.squeeze(0).to(DEVICE))
        torch.save(features.detach().cpu(), '{}/vgg19_ft_{}.pt'.format(FT_DIRECTORY, video))
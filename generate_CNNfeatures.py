import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import numpy as np
from argparse import ArgumentParser
import pandas as pd


class VideoDataset(Dataset):
    def __init__(self, videos_dir, video_names, score):
        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.score = score

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name))
        video_score = self.score[idx]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        video_channel = video_data.shape[3]

        transformed_video = torch.zeros([video_length, video_channel,  video_height, video_width])
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame

        sample = {'video': transformed_video,
                  'score': video_score}

        return sample


class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                return features_mean


def get_features(video_data, frame_batch_size=64, device=torch.device('cuda:0')):
    extractor = ResNet50().to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
        while frame_end < video_length:
            batch = video_data[frame_start:frame_end].to(device)
            features_mean = extractor(batch)
            output = torch.cat((output, features_mean), 0)
            frame_end += frame_batch_size
            frame_start += frame_batch_size

        last_batch = video_data[frame_start:video_length].to(device)
        features_mean = extractor(last_batch)
        output = torch.cat((output, features_mean), 0)
        output = output.squeeze()
    return output


if __name__ == "__main__":
    parser = ArgumentParser(description='Extracting high-level features')
    parser.add_argument('--database', type=str, default='KoNViD-1k')
    parser.add_argument('--frame_batch_size', type=int, default=32)
    parser.add_argument('--gpu_device', type=str, default='cuda:1')
    args = parser.parse_args()

    if args.database == 'KoNViD-1k':
        args.videos_dir = '../../database/KoNViD_1k_videos/'
        args.features_dir = 'data/CNN_features_KoNViD-1k/'
        info = pd.read_csv('data/KoNViD_1k_attributes.csv')
        file_names = info['flickr_id'].values
        file_names = [str(k)+'.mp4' for k in file_names]
        mos = info['MOS'].values

        args.video_names = file_names
        args.mos = mos

    if not os.path.exists(args.features_dir):
        os.makedirs(args.features_dir)

    args.device = torch.device(args.gpu_device)

    dataset = VideoDataset(args.videos_dir, args.video_names, args.mos)

    for i in range(len(dataset)):
        current_data = dataset[i]
        current_video = current_data['video']
        current_score = current_data['score']
        print('Video {}: length {}'.format(i, current_video.shape[0]))
        features = get_features(current_video, args.frame_batch_size, device=args.device)
        np.save(args.features_dir + str(i) + '_resnet-50_res5c', features.to('cpu').numpy())
        np.save(args.features_dir + str(i) + '_score', current_score)
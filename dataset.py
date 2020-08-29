from torch.utils.data import Dataset
import skvideo.io
import os
import numpy as np
import torch


class VideoFeatureDataset(Dataset):
    def __init__(self, idx_list, database_info, frame_sample_n=100):
        super(VideoFeatureDataset, self).__init__()
        video_name = database_info['video_name']
        mos = database_info['mos']
        video_dir = database_info['video_dir']
        feature_dir = database_info['feature_dir']

        mos = mos / mos.max()
        mos = mos.astype(np.float32)

        self.mos = mos[idx_list]
        self.video_name = video_name[idx_list]
        self.video_dir = video_dir
        self.feature_dir = feature_dir
        self.idx_list = idx_list
        self.frame_sample_n = frame_sample_n

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        video_data = skvideo.io.vread(os.path.join(self.video_dir, self.video_name[idx]))

        # video
        video_data = torch.from_numpy((video_data/255.0).astype(np.float32))
        video_data = video_data.permute([3, 0, 1, 2])

        # feature
        feature_id = self.idx_list[idx]
        feature_id_file = self.feature_dir + str(feature_id) + '_resnet-50_res5c' + '.npy'
        feature = np.load(feature_id_file)

        frames = video_data.shape[1]
        frame_idx_sel = np.linspace(start=0, stop=frames-1, num=self.frame_sample_n, dtype=np.int16)

        video_data = video_data[:, frame_idx_sel]
        feature = feature[frame_idx_sel]

        mos = self.mos[idx]

        sample = {'video': video_data,
                  'feature': feature,
                  'score': mos}
        return sample

import torch
from argparse import ArgumentParser
import pandas as pd
from SIONR import SIONR
from dataset import VideoFeatureDataset
import numpy as np
from fit_function import fit_function
from scipy import stats

parser = ArgumentParser(description='NR-VQA')
parser.add_argument('--gpu_device', type=str, default='cuda:2')
parser.add_argument('--database', type=str, default='KoNViD-1k')
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=3)
args = parser.parse_args()

if args.database == 'KoNViD-1k':
    video_dir = '../../database/KoNViD_1k_videos/'
    feature_dir = 'data/CNN_features_KoNViD-1k/'
    info = pd.read_csv('data/KoNViD_1k_attributes.csv')
    file_names = info['flickr_id'].values
    video_name = [str(k) + '.mp4' for k in file_names]
    video_name = np.array(video_name)
    mos = info['MOS'].values
    database_info = {'video_dir': video_dir,
                     'feature_dir': feature_dir,
                     'video_name': video_name,
                     'mos': mos}
    args.database_info = database_info
    scale = mos.max()
    args.scale = scale

split_idx_file = 'data/train_val_test_split.xlsx'
split_info = pd.read_excel(split_idx_file)
idx_all = split_info.iloc[:, 0].values
split_status = split_info['status'].values
test_idx = idx_all[split_status == 'test']

device = torch.device(args.gpu_device)

VQA_model = SIONR()
model_file = 'model/SIONR.pt'
VQA_model.load_state_dict(torch.load(model_file))
VQA_model = VQA_model.to(device)

test_dataset = VideoFeatureDataset(idx_list=test_idx, database_info=args.database_info)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size,
                                          num_workers=args.num_workers)

# test
y_predict = np.zeros(len(test_idx))
y_label = np.zeros(len(test_idx))
loss_sum = 0
with torch.no_grad():
    for i, data in enumerate(test_loader):
        video = data['video'].to(device)
        feature = data['feature'].to(device)
        label = data['score']
        label = label.to(device)
        y_label[i] = label.item()

        outputs = VQA_model(video, feature)

        y_predict[i] = outputs.item()

        print('test: ', i)

y_predict = fit_function(y_predict, y_label)
test_PLCC = stats.pearsonr(y_predict, y_label)[0]
test_SROCC = stats.spearmanr(y_predict, y_label)[0]
test_RMSE = np.sqrt((((y_predict - y_label) * args.scale) ** 2).mean())

result_excel = 'result/test_result.xlsx'
result = pd.DataFrame()
result['PLCC'] = [test_PLCC]
result['SROCC'] = [test_SROCC]
result['RMSE'] = [test_RMSE]
result.to_excel(result_excel)

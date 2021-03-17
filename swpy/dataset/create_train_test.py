import os
import argparse
from glob import glob
from datetime import datetime
import shutil

import numpy as np
import pandas as pd

# TODO: condig で data_dir を設定する
#######################################
# argparse
#######################################
parser = argparse.ArgumentParser()
parser.add_argument('--targets', dest='targets', default='flow_speed,BZ_GSM,F,proton_density',
                    help='acceptable physical quantities are "flow_speed", "BZ_GSM", "F", proton_density, "T", "Vx". ')

args = parser.parse_args()

pq_targets = args.targets.split(',')
pq_targets_str = '-'.join(pq_targets)

root_data_dir = os.environ['ROOT_DATA_DIR']
#######################################
# データ取得元 & データ作成先定義
#######################################
kyoto_dir = root_data_dir + 'kyoto/'
dst_dir  = kyoto_dir + 'DST/'

dst_event_dir = root_data_dir + 'swpy/storm_datasets/Gruet_2018/DST/'
omni_event_dir = root_data_dir + 'swpy/storm_datasets/Gruet_2018/omni_5min/%s/' % (pq_targets_str)

train_output_dir      = root_data_dir + 'swpy/storm_datasets/Gruet_2018/train/'
train_dst_output_dir  = train_output_dir + 'DST/'
train_omni_output_dir = train_output_dir + 'omni_5min/%s/' % (pq_targets_str)

os.makedirs(train_dst_output_dir, exist_ok=True)
os.makedirs(train_omni_output_dir, exist_ok=True)

test_output_dir       = root_data_dir + 'swpy/storm_datasets/Gruet_2018/test/'
test_dst_output_dir   = test_output_dir + 'DST/'
test_omni_output_dir  = test_output_dir + 'omni_5min/%s/' % (pq_targets_str)

os.makedirs(test_dst_output_dir, exist_ok=True)
os.makedirs(test_omni_output_dir, exist_ok=True)

# event定義ファイル
events_file= root_data_dir + 'swpy/storm_datasets/Gruet_2018/magnetic_storm_events.csv'

#============================
#============================
events_df = pd.read_csv(events_file)
train_events_df = events_df[events_df['use_data'] == 'train']
test_events_df = events_df[events_df['use_data'] == 'test']


dst_events  = glob(dst_event_dir + '*')
omni_events = glob(omni_event_dir + '*')
dst_events_set  = set(dst_events)
omni_events_set = set(omni_events)

# trainの場合
for index, event in train_events_df.iterrows():
    Data_start = event.Data_start
    Data_end   = event.Data_end
    start_dt = datetime.strptime(Data_start, '%Y-%m-%d %H:%M')
    end_dt   = datetime.strptime(Data_end, '%Y-%m-%d %H:%M')
    start_end_string = start_dt.strftime('%Y%m%d%H') + '_' + end_dt.strftime('%Y%m%d%H')

    dst_full_path  = dst_event_dir  + 'storm_DST_' + start_end_string + '.csv'
    omni_full_path = omni_event_dir + 'storm_OMNI_' + pq_targets_str + '_' + start_end_string + '.csv'

    if dst_full_path in dst_events_set:
        shutil.copyfile(dst_full_path, train_dst_output_dir + os.path.basename(dst_full_path))
        shutil.copyfile(omni_full_path, train_omni_output_dir + os.path.basename(omni_full_path))


# testの場合
for index, event in test_events_df.iterrows():
    Data_start = event.Data_start
    Data_end   = event.Data_end
    start_dt = datetime.strptime(Data_start, '%Y-%m-%d %H:%M')
    end_dt   = datetime.strptime(Data_end, '%Y-%m-%d %H:%M')
    start_end_string = start_dt.strftime('%Y%m%d%H') + '_' + end_dt.strftime('%Y%m%d%H')

    dst_full_path  = dst_event_dir  + 'storm_DST_' + start_end_string + '.csv'
    omni_full_path = omni_event_dir + 'storm_OMNI_' + pq_targets_str + '_' + start_end_string + '.csv'

    if dst_full_path in dst_events_set:
        shutil.copyfile(dst_full_path, test_dst_output_dir + os.path.basename(dst_full_path))
        shutil.copyfile(omni_full_path, test_omni_output_dir + os.path.basename(omni_full_path))

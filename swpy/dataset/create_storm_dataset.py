import os
import argparse
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import pyspedas
from pytplot import get_data, del_data

#######################################
# argparse
#######################################
parser = argparse.ArgumentParser()
parser.add_argument('--interp', action='store_true')
parser.add_argument('--targets', dest='targets', default='flow_speed,BZ_GSM,F,proton_density',
                    help='acceptable physical quantities are "flow_speed", "BZ_GSM", "F", proton_density, "T", "Vx". ')

args = parser.parse_args()

interp = args.interp
pq_targets = args.targets.split(',')
pq_targets_str = '-'.join(pq_targets)

# -----------------------------------------------------------

root_data_dir = os.environ['ROOT_DATA_DIR']

#######################################
# データ取得元 & データ作成先定義
#######################################
kyoto_dir = root_data_dir + 'kyoto/'
dst_dir  = kyoto_dir + 'DST/'

dst_output_dir = root_data_dir + 'swpy/storm_datasets/Gruet_2018/DST/'
os.makedirs(dst_output_dir, exist_ok=True)

omni_out_dir = root_data_dir + 'swpy/storm_datasets/Gruet_2018/omni_5min/%s/' % (pq_targets_str)
os.makedirs(omni_out_dir, exist_ok=True)

# event定義ファイル
events_file= root_data_dir + 'swpy/storm_datasets/Gruet_2018/magnetic_storm_events.csv'

# ------------------------------------------------------------

def make_dst_event_df(start_dt, end_dt, dst_dir):
    '''
    イベント期間のDST指数DataFrame作成
    '''
    def get_DST_file(dt, dst_dir):
        tgt_dir = dst_dir + str(dt.year) + '/'
        glob_pattern = tgt_dir + 'DST_' + str(dt.year)+ str(dt.month).rjust(2, '0') + '*'
        return glob(glob_pattern)[0]

    # 必要なファイルを開けてつなげる (磁気嵐が最低1ヶ月以上のファイルが3つになる場合はまず存在しないと考える。)
    file_s = get_DST_file(start_dt, dst_dir)
    file_e = get_DST_file(end_dt, dst_dir)

    dst_df = pd.read_csv(file_s)
    if file_s != file_e:
        dst_df_end = pd.read_csv(file_e)
        dst_df = pd.concat([dst_df, dst_df_end])

    # Time をdatetime式に変更
    # start と endに含まれる 部分だけ抜き出し
    start_int = int(start_dt.strftime('%Y%m%d%H'))
    end_int   = int(end_dt.strftime('%Y%m%d%H'))

    dst_event_df = dst_df[(dst_df['Time'] >= start_int) & (dst_df['Time'] <= end_int)]
    dst_event_df = dst_event_df.reset_index(drop=True)

    def ut24to0nextday(ymdh_int):
        ymd  = str(ymdh_int)[0:8]
        hour = str(ymdh_int)[8:10]

        if hour == '24':
            dt_ymd = datetime.strptime(ymd, '%Y%m%d')
            dt_ymd = dt_ymd + timedelta(1)
            return int(dt_ymd.strftime('%Y%m%d%H'))
        else:
            return ymdh_int

    dst_event_df['Time'] = dst_event_df['Time'].apply(ut24to0nextday)

    return dst_event_df

# ---

def output2csv_event_df(event_df, start_dt, end_dt, output_dir='./', data_label=''):
    '''
    DataFrame の CSV出力用関数
    '''
    output_path_base = output_dir + 'storm_'
    if data_label != '':
        output_path_base += data_label + '_'
    output_path = output_path_base + start_dt.strftime('%Y%m%d%H') + '_' + end_dt.strftime('%Y%m%d%H') + '.csv'
    event_df.to_csv(output_path, index=False)

# ---

# =========================
# DST データ作成
# =========================
event_set_df = pd.read_csv(events_file)

for intdex, event in event_set_df.iterrows():
    Data_start = event.Data_start
    Data_end   = event.Data_end

    start_dt = datetime.strptime(Data_start, '%Y-%m-%d %H:%M')
    end_dt   = datetime.strptime(Data_end, '%Y-%m-%d %H:%M')

    dst_event_df = make_dst_event_df(start_dt, end_dt, dst_dir)

    # ファイルを出力
    output2csv_event_df(dst_event_df, start_dt, end_dt, dst_output_dir, 'DST')


# =========================
# OMNI データ作成
# =========================

import pdb

convert_time = np.frompyfunc(lambda x: pyspedas.time_string(x, '%Y%m%d%H%M'), 1, 1)

event_set_df = pd.read_csv(events_file)
for intdex, event in event_set_df.iterrows():
    Data_start = event.Data_start
    Data_end   = event.Data_end

    start_dt = datetime.strptime(Data_start, '%Y-%m-%d %H:%M')
    end_dt   = datetime.strptime(Data_end, '%Y-%m-%d %H:%M')

    del_data()
    pyspedas.omni.data([Data_start, Data_end], datatype='5min')

    omni_event_df = pd.DataFrame()

    for target in pq_targets:
        time = get_data(target)[0]
        pq   = get_data(target)[1]

        omni_event_df['Time'] = convert_time(time)
        omni_event_df[target] = pq

    if interp:
        omni_event_df = omni_event_df.interpolate()

    output2csv_event_df(omni_event_df, start_dt, end_dt, omni_out_dir, 'OMNI_' + pq_targets_str)

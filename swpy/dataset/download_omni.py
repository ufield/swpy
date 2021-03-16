import argparse
import pyspedas


parser = argparse.ArgumentParser()
parser.add_argument('--start', dest='start_date', default='2000-01-01', help='start date of omni data (format=YYYY-mMM-DD).')
parser.add_argument('--end', dest='end_date', default='2020-12-31', help='end date of omni data (format=YYYY-mMM-DD).')

args = parser.parse_args()

s = args.start_date
e = args.end_date

pyspedas.omni.data([s, e], datatype='5min')
pyspedas.omni.data([s, e], datatype='1min')

import os
import sys

PROJECT_ROOT = '/'.join(os.path.realpath(__file__).split('/')[:-1])   # '/liaoweiduo/URL'
print(f'PROJECT_ROOT: {PROJECT_ROOT}')
META_DATASET_ROOT = os.environ['META_DATASET_ROOT']
print(f'META_DATASET_ROOT: {META_DATASET_ROOT}')
META_RECORDS_ROOT = os.environ['RECORDS']
print(f'META_RECORDS_ROOT: {META_RECORDS_ROOT}')
META_DATA_ROOT = '/'.join(META_RECORDS_ROOT.split('/')[:-1])
print(f'META_DATA_ROOT: {META_DATA_ROOT}')

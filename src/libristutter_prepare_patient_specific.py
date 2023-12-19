import os
import csv
import json
import string
import logging
import torchaudio
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio
from prepare_utils import create_manifest, check_folder
logger = logging.getLogger(__name__)
SAMPLE_RATE = 22050
import os
import numpy
import pandas
import random
import argparse
import numpy as np

def prepare_libristutter_patient(
    data_folder,
    train_manifest, valid_manifest, test_manifest,
    speaker, nb_files = 500, n_fold=10, isSiamese=False, valid_perc=0.1, test_perc=0.1
):
    # Make sure a sample of the files are correct
    if not check_folder(data_folder):
        raise ValueError(f"{data_folder} doesn't contain LibriStutter")
    # Collect files. Include dir separators to prevent spurious matches
    df = pandas.read_csv('/LibriStutter/LibriStats.csv')
    new_df = df[df['SPEAKER']==speaker]

    all_files = []
    for c,f in zip(new_df["CHAPTER"],new_df["FILE"]):
        all_files.append(f"{data_folder}/Audio/{speaker}/{c}/{f}")
    random.shuffle(all_files)
    all_files = all_files[:nb_files]

    # 5-fold CV
    for i in range(n_fold):
        all_files = numpy.roll(all_files, int(len(all_files)/n_fold))
        valid_files_cnt = int(len(all_files)*valid_perc)
        test_files_cnt = int(len(all_files)*test_perc)
        valid_filelist = all_files[-(valid_files_cnt+test_files_cnt):-test_files_cnt]
        test_filelist = all_files[-test_files_cnt:]
        train_filelist = list(set(all_files) - set(valid_filelist) - set(test_filelist))

        # Create json files
        list_of_stutters = []
        create_manifest(f'{train_manifest}_{isSiamese}_{i}', train_filelist, list_of_stutters, isSiamese)
        create_manifest(f'{valid_manifest}_{isSiamese}_{i}', valid_filelist, list_of_stutters, isSiamese)
        create_manifest(f'{test_manifest}_{isSiamese}_{i}', test_filelist, list_of_stutters, isSiamese)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', dest='data_folder', type=str, help='Original data path')
    parser.add_argument('--train_manifest', dest='train_manifest', type=str, help='json file for train set')
    parser.add_argument('--valid_manifest', dest='valid_manifest', type=str, help='json file for valid set')
    parser.add_argument('--test_manifest', dest='test_manifest', type=str, help='json file for test set')
    parser.add_argument('--nb_files', dest='nb_files', type=int, help='# of files to use')
    parser.add_argument('--n_fold', dest='n_fold', type=int, help='# of folds for K-Fold CV')
    parser.add_argument('--isSiamese', dest='isSiamese', type=int, help='Prepare for Siamese Network')
    parser.add_argument('--speaker_id', dest='speaker', type=int, help='Speaker ID')
    parser.add_argument('--valid_perc', dest='valid_perc', type=float, help='Split size of valid set')
    parser.add_argument('--test_perc', dest='test_perc', type=float, help='Split size of test set')

    args = parser.parse_args()
    prepare_libristutter_patient(
        args.data_folder,
        args.train_manifest, args.valid_manifest, args.test_manifest,
        speaker = args.speaker,
        nb_files = args.nb_files,
        n_fold = args.n_fold,
        isSiamese = args.isSiamese,
        valid_perc = args.valid_perc, test_perc = args.test_perc
    )
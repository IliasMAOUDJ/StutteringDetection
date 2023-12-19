import logging
import pandas
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio
from prepare_utils import check_folder, create_manifest
import random
import argparse

logger = logging.getLogger(__name__)
SAMPLE_RATE = 22050

import numpy
def prepare_libristutter_population(
    data_folder,
    train_manifest,
    valid_manifest,
    test_manifest,
    nb_files=500, n_fold=10, isSiamese=False, valid_perc=0.1, test_perc=0.1
):

    "Create the .json manifest files necessary for DynamicItemDataset"
    SPEAKERS = [x+1 for x in range(42)]
    # Make sure a sample of the files are correct
    if not check_folder(data_folder):
        raise ValueError(f"{data_folder} doesn't contain LibriStutter")



    # Collect files. Include dir separators to prevent spurious matches
    df = pandas.read_csv('./LibriStats.csv')
    new_df = df[df['SPEAKER'].isin(SPEAKERS)]
    all_files = []
    for s,c,f in zip(new_df["SPEAKER"],new_df["CHAPTER"],new_df["FILE"]):
        all_files.append(f"{data_folder}/Audio/{s}/{c}/{f}")
    random.shuffle(all_files)
    all_files = all_files[:nb_files]
    for i in range(n_fold):
        all_files = numpy.roll(all_files, int(len(all_files)/n_fold))
        print(len(all_files))
        valid_files_cnt = int(len(all_files)*valid_perc)
        test_files_cnt = int(len(all_files)*test_perc)
        valid_filelist = all_files[-(valid_files_cnt+test_files_cnt):-test_files_cnt]
        test_filelist = all_files[-test_files_cnt:]
        train_filelist = list(set(all_files) - set(valid_filelist) - set(test_filelist))
        print(valid_files_cnt, test_files_cnt)
        print(len(valid_filelist), len(train_filelist), len(test_filelist))
        list_of_stutters=[]
        # Create json files
        create_manifest(f'{train_manifest}', train_filelist, list_of_stutters, isSiamese)
        create_manifest(f'{valid_manifest}', valid_filelist, list_of_stutters, isSiamese)
        create_manifest(f'{test_manifest}', test_filelist, list_of_stutters, isSiamese)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default="/home/ilias-m/Documents/GitHub/LibriStutter/LibriStutter",dest='data_folder', type=str, help='Original data path')
    parser.add_argument('--train_manifest', default="train", dest='train_manifest', type=str, help='json file for train set')
    parser.add_argument('--valid_manifest', default="valid", dest='valid_manifest', type=str, help='json file for valid set')
    parser.add_argument('--test_manifest', default="test", dest='test_manifest', type=str, help='json file for test set')
    parser.add_argument('--nb_files', default=20000, dest='nb_files', type=int, help='# of files to use')
    parser.add_argument('--n_fold', default=1, dest='n_fold', type=int, help='# of folds for K-Fold CV')
    parser.add_argument('--isSiamese', default=0,dest='isSiamese', type=int, help='Prepare for Siamese Network')
    parser.add_argument('--valid_perc', default=0.1, dest='valid_perc', type=float, help='Split size of valid set')
    parser.add_argument('--test_perc', default=0.1,dest='test_perc', type=float, help='Split size of test set')

    args = parser.parse_args()
    prepare_libristutter_population(
        args.data_folder,
        args.train_manifest, args.valid_manifest, args.test_manifest,
        args.nb_files,
        args.n_fold,
        args.isSiamese,
        args.valid_perc, args.test_perc
    )
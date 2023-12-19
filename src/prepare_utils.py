import os 
from tqdm import tqdm
import torchaudio
import random
import csv
import pydub
import json
ALLOWED_UTT = ". "
# The indices correspond to the number in the annotation.
STUTTER_TYPE = [
    ".", "interjection", "sound_rep", "word_rep", "phrase_rep", "prolongation"
]

def check_folder(data_folder):
    "Check for main folders to exist"
    if not (
        os.path.isdir(data_folder)
        and os.path.isdir(os.path.join(data_folder, "Annotations"))
        and os.path.isdir(os.path.join(data_folder, "Audio"))
    ):
        return False
    return True

def compute_annotations(rel_path, length, annotation_path, list_of_stutters, i, isSiamese):
    "Read annotations from file and create annotations list. May involve chunking."

    annotation_types = "speaker", "breaks", "stutter_type"
    annotation = {t: [] for t in annotation_types}
    filename = os.path.basename(rel_path).split('/')[-1]
    speaker_id = filename.split('-')[0]

    annotation["speaker"]= speaker_id
    annotation["length"] = length
    annotation["wav"] = rel_path
    with open(annotation_path) as annotation_csv:
        csv_reader = csv.reader(annotation_csv)
        chunk_start = 0.0
        for row in csv_reader:
            if row[0] == "STUTTER":
                stutter = STUTTER_TYPE[int(row[3])]
                list_of_stutters.append(stutter)
                annotation["stutter_type"].append(stutter)
            else: 
                annotation["stutter_type"].append(".")
            word_start = float(row[1]) - chunk_start
            # Make these have only two decimal places
            break_time = "{:.2f}".format(word_start)
            annotation["breaks"].append(break_time)
        # Append final break (end of last word)
        word_end_time = float(row[2]) - chunk_start
        annotation["breaks"].append("{:.2f}".format(word_end_time))
    annotation["contain_stutter"] = not all(ch in ALLOWED_UTT for ch in annotation["stutter_type"])

    #if all utterances are "." then audio is "tested" (does not contain stutter)
    if(isSiamese):
        #if(annotation["contain_stutter"]):
        #    split = "ref"
        #else:
            if(i%2==0):
                split = "ref"
            else:
                split = "pair"
    else:
        split = "ref"

    for key in annotation_types:
        if key=="speaker":
            continue
        annotation[key] = " ".join(annotation[key])
    return annotation, split

import pandas as pd
def create_manifest(json_file, filelist, list_of_stutters, isSiamese):
    "Read all files and create manifest file with annotations"
    manifest_ref = {}
    manifest_pair = {}
    i=0
    for filepath in tqdm(filelist):
        # Load to compute length for sorting etc.
        audio, rate = torchaudio.load(filepath)
        length = audio.size(1) / rate
        #if(length<seg_len):
        #    continue
        # Split path into folders
        path_parts = filepath.split(os.path.sep)
        uttid = path_parts[-1][:-len(".flac")]
        
        rel_path = os.path.join("{data_root}", *path_parts[-4:])
        # Compute annotation path
        annotation_path_parts = path_parts.copy()
        annotation_path_parts[-1] = uttid + ".csv"
        annotation_path_parts[-4] = "Annotations"
        annot_path = os.path.sep + os.path.join(*annotation_path_parts)

        # Create entry
        #random.uniform(0, 1)
        annotation, split = compute_annotations(rel_path, length, annot_path, list_of_stutters, i, isSiamese)
        if(split=="pair"):
            manifest_pair[f"{uttid}"] = annotation
        else:
            manifest_ref[f"{uttid}"] = annotation
        i+=1

    # Write annotations to file
    with open(f'{json_file}_libri.json', "w") as w:
        json.dump(manifest_ref, w, indent=2)
    if(isSiamese):
        with open(f'{json_file}_pair.json', "w") as w:
            json.dump(manifest_pair, w, indent=2)
    print(f"Finished writing manifest in file {json_file}")

"""
def split_audio(file):
    with open(file) as f_in:
        f = json.load(f_in)
        id=0
        for audio in tqdm(f):
            ranges =[]
            
            all_lines = []
            speaker_id = audio.split("/")[-1].split("-")[0]
            book_id = audio.split("/")[-1].split("-")[1]
            chapter_id = audio.split("/")[-1].split("-")[2]
            filename = audio.split(".")[0]
            name = f"{speaker_id}-{book_id}-{chapter_id}"
            sound_file = pydub.AudioSegment.from_wav(
    f"/home/ilias-m/Documents/DATASETS/DATABRASE/Audio_Datasets/LibriStutter_all/Audio/{speaker_id}/{book_id}/{name}.flac").set_frame_rate(16000).split_to_mono()
            annot_file = f"/home/ilias-m/Documents/DATASETS/DATABRASE/Audio_Datasets/LibriStutter_all/Annotations/{speaker_id}/{book_id}/{name}.csv"
            df = pd.read_csv(annot_file, header=None)
            stutter = df[df.iloc[:,3]!=0]
            stutter = stutter.reset_index()
            print("\n\n")
            print(stutter)
            print("\n\n")
            start = stutter.iloc[:,2]
            print(start)
            stutter_type = stutter.iloc[:,-1]
            print(stutter_type)
            
            ranges.append((start,(start+10.24)))
            #all_lines.append((filename, len(ranges)-1,l[0],l[1],l[2], id))
            id += 1
            # milliseconds in the sound track
            for i, (x, y) in enumerate(ranges):
                new_file=sound_file[0][int(x*1000) : int(y*1000)]
                new_file.export(f"Clips_Libri_10sec/{name}_{i}.wav", format="wav", bitrate="256K")

    with open("annotation_10.csv", 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('speakerID','bookID','chapterID','start', 'end', 'stutter_type', 'ID'))
        writer.writerows(all_lines)
"""
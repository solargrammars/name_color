import json
import ipdb
import argparse

import string
import re

from os.path import join
from datetime import datetime
from random import shuffle
from settings import DATA_PATH
from color_utils import rgb2lab

parser = argparse.ArgumentParser(description="Process data from CL")
parser.add_argument("--data_path", type=str, default=DATA_PATH)
parser.add_argument("--train_size", type=float, default=0.9)
args = parser.parse_args()

def clean_text(text):
    if text is None: return None
    text = text.lower()
    valid_characters = string.ascii_letters + " " + "."
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    if all(c in valid_characters for c in text): return text
    else: return None

if __name__ == "__main__":
    
    print(str(datetime.now()), "Loading data")
    with open(join(args.data_path, "colors_raw.json"), "r") as f:
        data = json.load(f)
    
    print("Number of initial (name, color) pairs", len(data))

    print(str(datetime.now()), "Processing data")
    
    resulting_set = {}
    for color_id, color_data in data.items():

        color_name = clean_text(color_data["title"])

        if color_name is not None and  len(color_name) > 0:
            lab_color = rgb2lab( color_data["r"], color_data["g"], color_data["b"])
            resulting_set[color_name] = {
                    "name":[ ch for ch in color_name], 
                    "lab": lab_color, 
                    "rgb": (color_data["r"], color_data["g"], color_data["b"]) 
                    }


    print("Number of resulting instances ", len(resulting_set))
    
    resulting_list = [r for r  in resulting_set.values()] 
    shuffle(resulting_list)
    limit = int(args.train_size * len(resulting_list))
    train = resulting_list[:limit]
    test = resulting_list[limit:]
    
    print("length train", len(train))
    print("length test", len(test))

    print("Saving to disk")
    with open(join(args.data_path,"colors_train.jsonl"), "w") as f:
        for r in train:
            json.dump(r, f)
            f.write("\n")

    with open(join(args.data_path,"colors_test.jsonl"), "w") as f:
        for r in test:
            json.dump(r, f)
            f.write("\n")


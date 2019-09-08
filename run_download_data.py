import time
import ipdb
import json
import argparse
from os.path import join
from datetime import datetime
from colourlovers import ColourLovers

from settings import DATA_PATH
parser = argparse.ArgumentParser(description="Download data from CL")
parser.add_argument("--data_path", type=str, default=DATA_PATH)
parser.add_argument("--n_colors", type=int, default=1000000)
parser.add_argument("--chunk_size", type=int, default=100)
parser.add_argument("--sleep_time", type=int, default=1)
args = parser.parse_args()

def get_data(data_path, n_colors, chunk_size, sleep_time):
    """
    check http://www.colourlovers.com/api
    https://github.com/elbaschid/python-colourlovers
    """
    cl = ColourLovers()
    data = {} 
    print(str(datetime.now()), "Begin download")
    for i in range(1, n_colors, chunk_size):
        if i % 100 == 0:
            print(str(datetime.now()), i)
        time.sleep(sleep_time)
        col = cl.colors(num_results=chunk_size, result_offset=i)
        for c in col:
            data[c.id] = {
                    "title": c.title, 
                    "r": c.rgb.red,
                    "g": c.rgb.green,
                    "b": c.rgb.blue
                    }
    
    print("Saving to disk")
    with open(join(data_path,"colors_raw.json"), "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    get_data(args.data_path, args.n_colors, args.chunk_size, args.sleep_time)

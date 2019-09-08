import json
import itertools
import numpy as np
import collections
#import seaborn as sns
import matplotlib.pyplot as plt
import ipdb

#import matplotlib
from os.path import join, isfile
from IPython.display import HTML
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from settings import DATA_PATH, OUTPUT_PATH
from adjustText import adjust_text 
#https://adjusttext.readthedocs.io/en/latest/
from matplotlib.animation import FuncAnimation
#sns.set(color_codes=True)

from color_utils import lab2rgb, get_color_vis,get_incremental_color_vis


path_output = join(OUTPUT_PATH, "1559981836/generated_output_incremental.json")
path_test   = join(DATA_PATH, "/home/pablo/data/name-color/colors_test.jsonl")

consolidated_path = join(OUTPUT_PATH, "consolidated.json")


def get_consolidated():

    if isfile(consolidated_path):  
        with open(  join(OUTPUT_PATH, "consolidated.json") , "r" ) as f:
            consolidated = json.load(f)
        
        return consolidated
        
    else:
        with open(path_output, "r") as f:
            data_output = json.load(f)

        with open( path_test, "r" ) as f:
            data_test = [ json.loads(l) for  l in f.readlines()]

        name_color_dict = {"".join([ i for i in  d["name_in_chars"] if i !="<PAD>"]) : d["generated_color"] 
                               for d in data_output}

        consolidated = {}
        for d in data_test:    
            temp = []
            for i in range(len(d["name"])):
                lab_color = name_color_dict[ "".join(d["name"][:i+1]) ]
                rgb_color = lab2rgb(lab_color[0], lab_color[1], lab_color[2])
                rgb_color = [int(i*255) for i in rgb_color] 

                tup = ("".join(d["name"][:i+1]) , rgb_color )#, hidden )
                temp.append( tup)
            consolidated["".join(d["name"])] = temp

        with open( join(OUTPUT_PATH, "consolidated.json") , "w") as f:
            json.dump(consolidated, f)
    
        return consolidated
    
def get_subset_consolidated(consolidated, keyword, num):
    subset = [ i for i in consolidated.keys() if keyword in i][:num]
    subset_consolidated = []
    for i in subset:
        subset_consolidated.extend(consolidated[i])
    
    return subset_consolidated
    
if __name__ == "__main__":
    
    
    # consolidated is a dictionary with key a color name
    # and with values the incremental substrings with their
    # associated rgb colors
    consolidated        = get_consolidated()
    subset_consolidated = get_subset_consolidated(consolidated, "light", 5)
    ipdb.set_trace()
   
    #subset_consolidated_strings, subset_consolidated_colors = list( zip(*subset_consolidated))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
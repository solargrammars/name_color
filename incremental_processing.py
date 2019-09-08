import sys
import json
import ipdb
import time
import itertools
import argparse
import numpy as np
import collections
import matplotlib.pyplot as plt

from pprint import pprint
from os.path import join, isfile

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from adjustText import adjust_text
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation

from settings import DATA_PATH, OUTPUT_PATH
from color_utils import lab2rgb, get_color_vis,get_incremental_color_vis

path_output = join(OUTPUT_PATH, "1559981836/generated_output_incremental_hidden.json")
path_test   = join(DATA_PATH, "/home/user/data/name-color/colors_test.jsonl")
consolidated_path = join(OUTPUT_PATH, "consolidated_hidden.json")

def get_consolidated():
    """ load the data """
    with open(path_output, "r") as f:
        data_output = json.load(f)    
    return data_output
    
def get_subset_consolidated(consolidated, keyword, num=10, where="a"):
    """ filter consolidated list by keyword"""
    
    if where == "a": # anywhere
        subset = [ i for i in consolidated.keys() 
                  if keyword in i][:num]
    elif where == "b": # begining
        subset = [ i for i in consolidated.keys() 
                  if keyword == i[:len(keyword)]][:num]
    elif where  == "e": # end
        subset = [ i for i in consolidated.keys() 
                  if keyword == i[len(keyword):]][:num]
    
    subset_consolidated = []
    for i in subset:
        subset_consolidated.extend(consolidated[i])
    
    return subset, subset_consolidated

def get_dim_reduction(list_of_vectors, method="pca"):
    """ apply PCA/ TSNE to a list of vectors 
        TODO:  finetune TSNE
    """
    
    if method == "pca":
        print("using PCA")
        reductor = PCA(n_components = 2)
    
    else:
        print("using TSNE")
        reductor = TSNE(n_components=2,
                        perplexity=30,
                        learning_rate=500,
                        init="pca",
                        verbose=1
                       )
        
    result = reductor.fit_transform(list_of_vectors)
    x = [i[0] for i in result] 
    y = [i[1] for i in result] 
    
    return x,y

def get_dict_char_ancestors(subset, consolidated):
    """
        for each element in subset, we need to link
        their substrings in a sequential way
        lets say , "pablo":
        p -> pa
        pa -> pab
        pab -> pabl
        pablo -> pabl
    """
    dict_char_ancestors = {}
    for color_name in subset:
        substrings = [ i["sub_name"] for i in consolidated[color_name]]
        for i, substr in enumerate(substrings):
            if  i == 0 :
                 dict_char_ancestors[substr] = [] 
            else: 
                if substr in dict_char_ancestors:
                    dict_char_ancestors[ substr] = dict_char_ancestors[substr] +  [substrings[i-1]]
                else:
                     dict_char_ancestors[substr] = [substrings[i-1]]
            
    return dict_char_ancestors  
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="name to color generation")
    parser.add_argument("--intermediate", action="store_true")
    parser.add_argument("--trajectory",   action="store_true")
    parser.add_argument("--persistname",  action="store_true")
    parser.add_argument("--justnames",  action="store_true")
    parser.add_argument("--keyword", type=str, default="red")
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--where", type=str, default="a")
    parser.add_argument("--redu", type=str, default="pca")
    
    args = parser.parse_args()
    pprint(vars(args))
    experiment_id = str(int(time.time())) 
    plain_args = "_".join([ k+":"+str(v) for k,v in   vars(args).items()])
    
    # consolidated is a dictionary with key a color name
    # and with values the incremental substrings with their
    # associated rgb colors
    print("loading consolidated")
    consolidated = get_consolidated()
    
    
    # if i just want to check the available names, nothing else
    
    if args.justnames == True:
        
        names = sorted(list(consolidated.keys()))
        pprint(names)
        sys.exit()
    
    
    # given a keyword , select colors names from consolidated
    # that contain such keyword
    print("obtaining subset")
    subset, subset_consolidated = get_subset_consolidated(
        consolidated, 
        args.keyword, 
        num=args.num,
        where=args.where)
    
    print("found ", len(subset) , " color names matching:")
    pprint(subset)
    
    
    # we split string from rgb vectors
    (subset_consolidated_strings, 
     subset_consolidated_colors_lab, 
     subset_consolidated_hidden, 
     subset_consolidated_colors_rgb)  = list( 
        zip(*[list(i.values()) for i in subset_consolidated]))
    

    # we transform vectos into 2-d representations
    colors_x, colors_y = get_dim_reduction(subset_consolidated_hidden, 
                                          method=args.redu)
    
    # associate substrings with their 2-d vectors
    substr_xy_dict = { subset_consolidated_strings[i]:(colors_x[i], colors_y[i]) 
                  for i in range(len(subset_consolidated_strings))}
     
    # get ancestors 
    dict_char_ancestors = get_dict_char_ancestors(subset, consolidated) 
    
    ## prepare for animate
    min_x = min(colors_x)
    max_x = max(colors_x)
    min_y = min(colors_y)
    max_y = max(colors_y)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_xlim(min_x -5, max_x +5)
    ax.set_ylim(min_y -5, max_y +5)
    #ax.set_axis_off()
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    
    step_text = ax.text(0.5, 0.9, '', transform=ax.transAxes)
    
    if args.intermediate == True:
        print("generating intermediate figure")
        plt.scatter(colors_x,
                    colors_y, 
                    s=50, 
                    c=np.array(subset_consolidated_colors_rgb)/255 )

        plt.savefig(join("output", experiment_id+"_"+"inter_"+plain_args+".png"))
        plt.close()
    
    else:
        print("generating animation")
        
        def animate(k):
            print("k", k , subset_consolidated_strings[k])
            #ax.set_title(subset_consolidated_strings[k], fontsize=20)
            
            step_text.set_text('{0}'.format(subset_consolidated_strings[k]))

            if args.trajectory == True: 
                if subset_consolidated_strings[k] in dict_char_ancestors:
                    for xx in dict_char_ancestors[subset_consolidated_strings[k]]:
                        # arrow between points
                        ax.arrow(
                            substr_xy_dict[xx][0], # origin x
                            substr_xy_dict[xx][1], # origin y

                            substr_xy_dict[subset_consolidated_strings[k]][0] - substr_xy_dict[xx][0], # dx
                            substr_xy_dict[subset_consolidated_strings[k]][1] - substr_xy_dict[xx][1], # dy

                            head_width = 0.2, 
                            head_length= 0.2,
                            length_includes_head = True,
                            color='k',
                            width=0.01,
                            alpha=0.05)
                    
            # add current point
            point_color = np.array(subset_consolidated_colors_rgb)[k]/255
            point_color = point_color.reshape(1,3)
            ax.scatter(colors_x[k],colors_y[k], s=100, c=point_color, alpha=0.8)

            
            # persist the final color name in the figure 
            if args.persistname == True:
                if  k+1 < len(subset_consolidated_strings) -1:
                    if len(subset_consolidated_strings[k+1]) ==1:
                        texts = [ax.text(colors_x[k], colors_y[k], 
                                subset_consolidated_strings[k], 
                                fontsize=10, ha='right', va='center')]

                        adjust_text(texts,expand_text=[2,2], expand_points=[2,2])
                        #adjust_text(texts)

        # define animation object
        ani = FuncAnimation(fig, 
                            animate, 
                            frames=len(subset_consolidated_strings), 
                            repeat=False, 
                            interval=200)
        
        print("saving animation")
                
        #ani.save(join("output", experiment_id+"_"+"ani_"+plain_args+".gif"),
        #         writer='imagemagick')
        
        writer=animation.FFMpegWriter(bitrate=1000, fps=5)
        ani.save(
            join("output", experiment_id+"_"+"ani_"+plain_args+".mp4"),
            writer=writer, #'ffmpeg',
           
        )
        print("finished, bye")
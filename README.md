# name-color

Transform a name or description into a color representation

This is the code used in [this article](https://solargrammars.github.io/blog/learning-color-language-representations.html) where 
the paper [Character Sequence Models for Colorful Words](https://aclweb.org/anthology/D16-1202)
is analyzed.

This code was tested on Python 3.6 / Pytorch 1.0

## How to use it

1. Modify `settings.py` according to your system.  

2. `run_download_data.py` will download tuples (color, description) from ColourLovers using their API.  

3. `run_process_data.py` will clean and split the data.  

4. `run_train.py` will train and save the best model.  

4. `run_inference.py` and `run_incremental_inference.py` are used to obtain the reuslts on the test set.  

## Acknowledgement

Most of the content on this repository is based on the great Pytorch tutorials by [epochx](https://github.com/epochx/pytorch-nlp-tutorial) 

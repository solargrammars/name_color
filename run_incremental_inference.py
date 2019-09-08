import torch
import json
import itertools
import ipdb
import numpy as np
import collections

from os.path import join
from settings import DATA_PATH
from data import get_incremental_test_set
from model import NameColor

from color_utils import lab2rgb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True

model_path = "/home/user/output/name-color/1559981836" # replace with real path 
with open(join(model_path,"params.json"), "r") as f:
    model_params = json.load(f)

print("loading test test")
test_iterator, vocab  = get_incremental_test_set(model_params["batch_size"])

INPUT_DIM = len(vocab)
PAD_IDX = vocab.PAD.hash

return_hidden = True

print("loading model")
model = NameColor(INPUT_DIM,
    model_params["embedding_dim"],
    model_params["hidden_dim"],
    model_params["output_dim"], 
    model_params["n_layers"],
    model_params["dropout"],
    PAD_IDX,
    return_hidden=return_hidden) # if we want to collect the hidden state 

model.load_state_dict(torch.load(
    join (model_path, "namecolor-model.pt")))
model.to(DEVICE)



def get_results(model, iterator, return_hidden):
    results = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            
            
            batch.to_torch_(DEVICE)
            text = batch.src_batch.sequences
            text_lengths = batch.src_batch.lengths
            
            
            print("batch ", i, " text size ", len(text)  )
            
            if return_hidden is False:
                predictions = model(text, text_lengths)
                predictions.squeeze(1)
            
                output = list(zip( 
                    text.cpu().numpy(), 
                    predictions.cpu().numpy()
                ))
            else:
                
                predictions, hidden = model(text, text_lengths)
                
                predictions.squeeze(1)
            
                output = list(zip( 
                    text.cpu().numpy().tolist(), 
                    predictions.cpu().numpy().tolist(),
                    hidden.cpu().numpy().tolist()
                ))
            
            results.append(output)
            
    return results

print("begin iteration")

result = get_results(model, test_iterator, return_hidden)

result_structured = {}
for res in result:
    key = "".join([i.string for i in vocab.indices2tokens(res[0][0]) if i.string != "<PAD>"])

    subs = []
    for sub in res[::-1]:
        sub_name = "".join([i.string for i in vocab.indices2tokens(sub[0]) if i.string != "<PAD>" ])
        sub_lab = sub[1]
        sub_rgb = lab2rgb(sub_lab[0], sub_lab[1], sub_lab[2])
        sub_rgb = [int(i*255) for i in sub_rgb]
        
        
        
        sub_hidden = sub[2] 
    
        subs.append(
            {
            "sub_name": sub_name,
            "sub_lab": sub_lab,
            "sub_hidden": sub_hidden,
            "sub_rgb":  sub_rgb
            }
        )
    result_structured[key] = subs
    
print("Saving to ", join(model_path, "generated_output_incremental_hidden.json"))
with open( join(model_path, "generated_output_incremental_hidden.json"), "w") as f:
    json.dump(result_structured, f)
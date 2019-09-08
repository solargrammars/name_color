import torch
import json
import itertools
import ipdb
import numpy as np
import collections

from os.path import join
from settings import DATA_PATH
from data import get_data
from model import NameColor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True

model_path = "/home/user/output/name-color/1559981836"  # replace with real path to model
with open(join(model_path,"params.json"), "r") as f:
    model_params = json.load(f)

_, test_iterator, vocab = get_data(
                    model_params["batch_size"], DEVICE)

INPUT_DIM = len(vocab)
PAD_IDX = vocab.PAD.hash

model = NameColor(INPUT_DIM,
    model_params["embedding_dim"],
    model_params["hidden_dim"],
    model_params["output_dim"], 
    model_params["n_layers"],
    model_params["dropout"],
    PAD_IDX)

model.load_state_dict(torch.load(
    join (model_path, "namecolor-model.pt")))
model.to(DEVICE)

def get_results(model, iterator):
    results = []
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            
            batch.to_torch_(DEVICE)
            text = batch.src_batch.sequences
            text_lengths = batch.src_batch.lengths
            predictions = model(text, text_lengths).squeeze(1)
            
            output = list(zip( 
                text.cpu().numpy(), 
                predictions.cpu().numpy(),
                batch.tgt_batch.sequences.cpu().numpy()
            ))
            results.extend(output)
            
    return results


result = get_results(model, test_iterator)

result_structured = [ 
    {
     "name_in_chars": [ i.string for  i in vocab.indices2tokens(name_in_ids)],
     "name_in_ids":  name_in_ids.tolist(),
     "generated_color":  gen_color.tolist(),
     "ground_truth_color": gt_color.tolist()   
        
    }  for name_in_ids, gen_color, gt_color in result ]


print("Saving to ", join(model_path, "generated_output.json") )
with open( join(model_path, "generated_output.json"), "w") as f:
    json.dump(result_structured, f)
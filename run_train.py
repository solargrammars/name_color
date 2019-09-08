import os
import json
import ipdb
import torch
import time
import argparse
from os.path import join
import torch.nn  as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from data import get_data
from model import NameColor
from settings import OUTPUT_PATH

SEED = 1234
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description="Name to Color")
parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--embedding_dim", type=int, default=300)
parser.add_argument("--hidden_dim", type=int, default=300)
parser.add_argument("--output_dim", type=int, default=3)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0001)

args = parser.parse_args()

def count_parameters(model):
        return sum(p.numel() 
                for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    model.train()
    for batch in iterator:
        
        batch.to_torch_(DEVICE)
        optimizer.zero_grad()
        text = batch.src_batch.sequences
        text_lengths = batch.src_batch.lengths
        predictions = model(text, text_lengths).squeeze(1)
        
        loss = criterion(predictions, batch.tgt_batch.sequences)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),0.25)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            batch.to_torch_(DEVICE)
            text = batch.src_batch.sequences
            text_lengths = batch.src_batch.lengths
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions , batch.tgt_batch.sequences)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == "__main__":
    
    train_iterator, test_iterator, vocab = get_data(
            args.batch_size, DEVICE)
    
    INPUT_DIM = len(vocab)
    PAD_IDX = vocab.PAD.hash

    model = NameColor(INPUT_DIM,
            args.embedding_dim, 
            args.hidden_dim, 
            args.output_dim, 
            args.n_layers,  
            args.dropout,
            PAD_IDX)

    print(model)
    
    

    model_id = str(int(time.time()))
    save_path = join(OUTPUT_PATH, model_id)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)

    with open(os.path.join(save_path,"params.json"), "w") as f:
            json.dump(vars(args), f)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)
    
    best_valid_loss = float('inf')

    print('Begin training')
    for epoch in range(args.n_epochs):
        start_time = time.time()
        train_loss = train(model, train_iterator, 
                optimizer, criterion)
        valid_loss  = evaluate(model, test_iterator, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), join(save_path,'namecolor-model.pt'))
        
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('valid_loss', valid_loss, epoch)
        
        print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} |')

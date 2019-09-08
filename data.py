

import torch
from torchtext import data
from os.path import join
from settings import DATA_PATH
import json
import ipdb
import numpy as np

UNK = '<UNK>'
PAD = '<PAD>'
BOS = '<BOS>'
EOS = '<EOS>'


def custom_tokenizer(text):
    return [ch for ch in text]


def get_incremental_test_set(batch_size):
    
    with open( join(DATA_PATH, "colors_train.jsonl" ), "r" ) as f:
        train_data = [ json.loads(l) for  l in f.readlines()]
        
    with open( join(DATA_PATH, "colors_test.jsonl" ), "r" ) as f:
        test_data = [ json.loads(l) for  l in f.readlines()]
    
    train_names = [i["name"] for i in  train_data]
    
    # we have the vocabulary
    vocab = Vocab(train_names, 
                  min_count= 1, 
                  add_padding=True)
    
    splitted = []
    for d in test_data[:1000]:
        if len(d["name"]) > 4:
            temp = []
            for i in range(len(d["name"])):
                temp.append(d["name"][:i+1])
            splitted.append(temp)
    
    #splitted_unique = list(set(tuple(x) for x in splitted))
    
    
    
    #splitted_unique_batches  = BatchIterator(vocab, 
    #                                         splitted_unique,
    #                                         batch_size, 
    #                                         ModelBatchIncremental)
    
    splitted_batches  = BatchIteratorVS(vocab, 
                                        splitted,
                                        ModelBatchIncremental)
    
    return splitted_batches, vocab

def get_data(batch_size, device):
    
    with open( join(DATA_PATH, "colors_train.jsonl" ), "r" ) as f:
        train_data = [ json.loads(l) for  l in f.readlines()]
        
    with open( join(DATA_PATH, "colors_test.jsonl" ), "r" ) as f:
        test_data = [ json.loads(l) for  l in f.readlines()]
    
    train_names = [i["name"] for i in  train_data]
    
    
    vocab = Vocab(train_names, 
                  min_count= 1, 
                  add_padding=True)
        
    train_batches = BatchIterator(vocab, train_data, batch_size, ModelBatch)
    test_batches  = BatchIterator(vocab, test_data,  batch_size, ModelBatch)
    
    return train_batches, test_batches, vocab
        
        
def tokenizer(x):
    #return list(x) 
    return x 


def token_function(x):
    #return x.lower() 
    return x

class VocabItem:
    def __init__(self, string, hash=None):
        self.string = string
        self.count = 0
        self.hash = hash 
        
    def __str__(self):
        return  'VocabItem({})'.format(self.string)

    def __repr__(self):
        return  self.__str__()
    
class Vocab:
    def __init__(self, sequences, tokenizer=tokenizer,
            token_function = token_function, 
            min_count=0, add_padding= False, add_bos=False,
            add_eos=False, unk=None):

        vocab_items= []
        vocab_hash = {}
        item_count = 0

        self.token_function = token_function
        self.tokenizer = tokenizer
        self.special_tokens = [] 
        
        self.UNK = None
        self.PAD = None 
        self.BOS = None
        self.EOS = None

        index2token = []
        token2index = {}


        for sequence in sequences :
            for item in tokenizer(sequence):
                real_item = token_function(item)
                if real_item not in vocab_hash:
                    vocab_hash[real_item] = len(vocab_items)
                    vocab_items.append(VocabItem(real_item))

                vocab_items[vocab_hash[real_item]].count +=1
                item_count +=1


                #if item_count % 100 == 0:
                #    print("Reading item {}".format(item_count))

        tmp = []
        if unk : 
            self.UNK = VocabItem(unk, hash=0)
            self.UNK.count = vocab_items[vocab_hash[unk]].count
            index2token.append(self.UNK)
            self.special_tokens.append(self.UNK)

            for token in vocab_items:
                if token.string != unk:
                    tmp.append(token)
        else:
            self.UNK = VocabItem(UNK, hash=0)
            index2token.append(self.UNK)
            self.special_tokens.append(self.UNK)

            for token in vocab_items:
                if token.count <= min_count:
                    self.UNK.count += token.count
                else:
                    tmp.append(token)

        tmp.sort(key=lambda token: token.count, reverse=True)

        if add_bos:
            self.BOS = VocabItem(BOS)
            tmp.append(self.BOS)
            self.special_tokens.append(self.BOS)

        if add_eos:
            self.EOS = VocabItem(EOS)
            tmp.append(self.EOS)
            self.special_tokens.append(self.EOS)
        
        if add_padding:
            self.PAD = VocabItem(PAD)   
            tmp.append(self.PAD)
            self.special_tokens.append(self.PAD)

        index2token += tmp
        for i, token in enumerate(index2token):
            token2index[token.string] = i
            token.hash = i

        self.index2token = index2token
        self.token2index = token2index

        print('Unknown vocab size:', self.UNK.count)
        print('Vocab size: %d' % len(self))

    def __getitem__(self, i):
        return self.index2token[i]
    def __len__(self):
        return len(self.index2token)
    def __iter__(self):
        return iter(self.index2token)
    def __contains__(self, key):
        return key in self.token2index

    def string2indices(self, string, add_bos=False, add_eos=False):

        string_seq = []
        if add_bos:
            string_seq.append(self.BOS.hash)
        for item in self.tokenizer(string):
            processed_token = self.token_function(item)
            string_seq.append(self.token2index.get(processed_token, self.UNK.hash))
        if add_eos:
            string_seq.append(self.EOS.hash)
        return string_seq

    def indices2tokens(self, indices, ignore_ids=()):

        tokens = []
        for idx in indices:
            if idx in ignore_ids:
                continue
            tokens.append(self.index2token[idx])

        return tokens

    
    
class BatchIterator(object):

    def __init__(self, vocabs, examples, batch_size, batch_builder,
                 shuffle=False, max_len=None):

        #self.mode = mode
        self.vocabs = vocabs
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.examples = examples
        self.num_batches = (len(self.examples) + batch_size - 1) // batch_size
        self.batch_builder = batch_builder

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        examples_slice = []
        for i, example in enumerate(self.examples, 1):
            examples_slice.append(example)
            if i > 0 and i % (self.batch_size) == 0:
                yield self.batch_builder(#self.mode,
                                         examples_slice,
                                         self.vocabs,
                                         max_len=self.max_len)
                examples_slice = []

        if examples_slice:
            yield self.batch_builder(#self.mode,
                                     examples_slice,
                                     self.vocabs,
                                     max_len=self.max_len)    

class BatchIteratorVS(object):

    def __init__(self, vocabs, examples, batch_builder,
                 shuffle=False, max_len=None):

        #examples is a list of lists, with each list a decomposition 
        # of a color name
        
        self.vocabs = vocabs
        self.max_len = max_len
        
        self.shuffle = shuffle
        self.examples = examples
        self.num_batches = len(examples)
        self.batch_builder = batch_builder

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        
        for example_slice in self.examples:
            
            yield self.batch_builder(
                                         example_slice,
                                         self.vocabs,
                                         max_len=self.max_len)
  

            
def pad_list(sequences, dim0_pad=None, dim1_pad=None,
             align_right=False, pad_value=0):
    """
    Receives a list of lists and returns a padded 2d ndarray,
    and a list of lengths. 
    
    sequences: a list of lists. len(sequences) = M, and N is the max
               length of any of the lists contained in sequences.
               e.g.: [[2,45,3,23,54], [12,4,2,2], [4], [45, 12]]
   
    Returns a numpy ndarray of dimension (M, N) corresponding to the padded
    sequences and a list of the original lengths.
    
    Returns:
       - out: a torch tensor of dimension (M, N) 
       - lengths: a list of ints containing the lengths of each element
                  in sequences
       
    """
    
    sequences = [np.asarray(sublist) for sublist in sequences]
    
    if not dim0_pad:
        dim0_pad = len(sequences)

    if not dim1_pad:
        dim1_pad = max(len(seq) for seq in sequences)

    out = np.full(shape=(dim0_pad, dim1_pad), fill_value=pad_value)

    lengths = []
    for i in range(len(sequences)):
        data_length = len(sequences[i])
        lengths.append(data_length)
        offset = dim1_pad - data_length if align_right else 0
        np.put(out[i], range(offset, offset + data_length), sequences[i])

    lengths = np.array(lengths)

    return out, lengths            

class Batch:
  
    # we add the four major components to our low level Batch object
    def __init__(self, sequences, lengths, sublengths, masks):
        
        self.sequences = sequences
        self.lengths = lengths
        self.sublengths = sublengths
        self.masks = masks
    
    # following PyTorch's approach, we add a method 
    # to transform this batch into a collection of PyTorch
    # objects in-place, and name it using an underscore at the end
    def to_torch_(self, device,  seq_require_float=False):
       
        
        if seq_require_float: 
            self.sequences = torch.tensor(self.sequences,
                                          device=device,
                                          dtype=torch.float)
        else:
            self.sequences = torch.tensor(self.sequences,
                                          device=device,
                                          dtype=torch.long)
       
        if self.lengths is not None:
            self.lengths = torch.tensor(self.lengths,
                                        device=device,
                                        dtype=torch.long)

        if self.sublengths is not None:
            self.sublengths = torch.tensor(self.sublengths,
                                           device=device,
                                           dtype=torch.long)
        
        # note we specify the dtupe of the object to float 
        if self.masks is not None:
            self.masks = torch.tensor(self.masks,
                                      device=device,
                                      dtype=torch.float)


            
class ModelBatch(object):

    def __init__(self, examples, vocab, max_len=None):

        examples = sorted(
            [ (i, len(i["name"])) for i in examples],
            key = lambda x:x[1], reverse=True)
        
        examples = [i[0] for i in examples]
    
        src_examples = [vocab.string2indices(sentence["name"][:max_len])
                        for sentence in examples]
      
        # we map the output labels to its corresponding ids
        tgt_examples = [sentence["lab"] for sentence in examples]

        # we pad our batches so they can be stored in a tensor
        # and recover the padded input and the original example lenghs 
        src_padded, src_lengths = pad_list(src_examples,
                                           pad_value=vocab.PAD.hash)
  
        
        self.src_batch = Batch(src_padded, src_lengths, None, None)

        self.tgt_batch = Batch(tgt_examples, None, None, None)
    
    def to_torch_(self, device):
        self.src_batch.to_torch_(device)
        self.tgt_batch.to_torch_(device, seq_require_float=True)            
            

class ModelBatchIncremental(object):

    def __init__(self, examples, vocab, max_len=None):

        #ipdb.set_trace()
        
        examples = sorted(
            [ (i, len(i)) for i in examples],
            key = lambda x:x[1], reverse=True)
        
        examples = [i[0] for i in examples]
        
        
        
        src_examples = [vocab.string2indices(sentence[:max_len])
                        for sentence in examples]
      
        # we map the output labels to its corresponding ids
        #tgt_examples = [sentence["lab"] for sentence in examples]

        # we pad our batches so they can be stored in a tensor
        # and recover the padded input and the original example lenghs 
        src_padded, src_lengths = pad_list(src_examples,
                                           pad_value=vocab.PAD.hash)
  
        
        self.src_batch = Batch(src_padded, src_lengths, None, None)

        #self.tgt_batch = Batch(tgt_examples, None, None, None)
    
    def to_torch_(self, device):
        self.src_batch.to_torch_(device)
        #self.tgt_batch.to_torch_(device, seq_require_float=True)               
            
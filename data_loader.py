import torch
import numpy as np
import json
import pandas as pd
import random
import os
import math

gan_examples = 4
channels = 11
max_size = 10
max_num_examples = 5


class Tagger:
    def __init__(self):
        self.tags = pd.read_csv('../data/tags.csv', index_col=0).transpose()

    def get_tag(self, id):
        return self.tags[id][self.tags[id] == 1].index.tolist()


def get_size(grid):
    return len(grid), len(grid[0])


def is_within_size_limit(data):
    ok = 0
    for set in data.values():
        for idx, pair in enumerate(set):
            for grid in pair.values():
                x, y = get_size(grid)
                if x <= max_size and y <= max_size:
                    ok += 0.5
            ok = math.floor(ok)

    if ok >= 3:
        return True

    return False


def make_onehot(grid, zero_pad=True):
    grid = torch.tensor(grid)
    if zero_pad is False:
        return torch.zeros(10, *grid.shape).scatter_(0, grid.unsqueeze(0), 1)
    else:
        if grid.shape[1] > max_size or grid.shape[0] > max_size:
            return None
        x, y = tuple(grid.shape)
        out = torch.zeros(10, x, y).scatter_(0, grid.unsqueeze(0), 1)
        pad_x = (max_size - x) // 2
        pad_y = (max_size - y) // 2
        zero_padded = torch.zeros(channels, max_size, max_size)  # last dim is out-of-bounds
        if channels == 11:
            zero_padded[-1, ...] = 1
            zero_padded[:, pad_x:pad_x + x, pad_y:pad_y + y] = torch.cat([out, torch.zeros_like(out)[0].unsqueeze(0)], dim=0)
        elif channels == 10:
            zero_padded[:, pad_x:pad_x + x, pad_y:pad_y + y] = out

        return zero_padded


def torchify(data, zero_pad=True):
    for idx, pair in enumerate(data['train']):
        data['train'][idx]['input'] = make_onehot(data['train'][idx]['input'], zero_pad)
        data['train'][idx]['output'] = make_onehot(data['train'][idx]['output'], zero_pad)
        
    for idx, pair in enumerate(data['test']):
        data['test'][idx]['input'] = make_onehot(data['test'][idx]['input'], zero_pad)
        data['test'][idx]['output'] = make_onehot(data['test'][idx]['output'], zero_pad)
        
        
    return data


def make_meta(data, zero_pad=True):
    out = {
        'train': [[], []],
        'test': [[], []]
    }

    pairs = []
    for k, v in data.items():
        for idx, pair in enumerate(v):
            for kk, vv in pair.items():
                if not isinstance(vv, torch.Tensor) and vv is not None:
                    pair[kk] = torch.tensor(vv)
            if None not in pair.values():
                pairs.append(pair)
    random.shuffle(pairs)
    if len(pairs) > max_num_examples + 1:
        pairs = pairs[:max_num_examples + 1]

    for pair in pairs[:-1]:
        for kk, vv in pair.items():
            out['train'][0 if kk == 'input' else 1].append(vv)

    num_examples = len(pairs) - 1
    if zero_pad:
        for _ in range(max_num_examples - num_examples):
            out['train'][0].append(torch.zeros(*out['train'][0][0].shape))
            out['train'][1].append(torch.zeros(*out['train'][1][0].shape))

    out['test'][0].append(pairs[-1]['input'])
    out['test'][1].append(pairs[-1]['output'])

    for k, v in out.items():
        for idx, vv in enumerate(v):
            out[k][idx] = torch.stack(vv, dim=0)
    return out


class Tasks(torch.utils.data.Dataset):
    def __init__(self, eval=False, embedded=False, tag=None, return_id=False):
        super().__init__()
        self.return_id = return_id
        self.embedded = embedded

        self.path = 'data/training/' if not eval else 'data/evaluation/'
        if embedded:
            self.path = 'data/embedded/ ' + ('training/' if not eval else 'evaluation/')
        ids = os.listdir(self.path)
        if tag is not None:
            if not isinstance(tag, list):
                tag = [tag]

            tags = pd.read_csv('data/tags.csv', index_col=0)  # contains a boolean table where columns are tasks and rows are IDs
            tagged_ids = []
            for t in tag:
                tagged_ids += tags[tags[t] == 1].index.to_list()  # selects those IDs matching the tag
            ids = list(set(tagged_ids).intersection(ids))

        self.ids = []
        for id in ids:
            data = json.load(open(self.path + id))
            if is_within_size_limit(data):
                self.ids.append(id)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]
        data = json.load(open(self.path + id))
        if self.embedded:
            out = make_meta(data)
        else:
            out = make_meta(torchify(data))
        if self.return_id:
            return id, out
        return out



class Grids_n(torch.utils.data.Dataset):
    def __init__(self, eval=False, ids=None):
        super().__init__()
        
        self.count = 0
        self.path = 'data/training/' if not eval else 'data/evaluation/'
        if ids is None:
            ids = os.listdir(self.path)

        self.grids = []
        self.ids = []
        for id in ids:
            data = json.load(open(self.path + id))
            data = torchify(data, zero_pad=True)
            for section in data.values():  # train and test sections
                for pair in section:
                    for grid in pair.values():  # input and output
                        
                        if grid is not None:
                            grid = grid.numpy().astype(np.bool)
                            self.grids.append(grid)
        self.grids = np.array(self.grids, dtype=np.bool)

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, item):
        return torch.from_numpy(self.grids[item]).float()

class Grids(torch.utils.data.Dataset):
    def __init__(self, eval=False, ids=None, tensor_type="float"):
        super().__init__()
       
        self.path = 'data/training/' if not eval else 'data/evaluation/'
        self.tensor_type = tensor_type
        if ids is None:
            ids = os.listdir(self.path)

        self.grids_inputs = []
        self.grids_outputs = []
        
        self.ids = []
        for id in ids:
            data = json.load(open(self.path + id))
            data = torchify(data, zero_pad=True)
            for section in data.values():  # train and test sections
                for pair in section:

                    input1 = pair['input']
                    output1 = pair['output']# input and output
                    
                    if input1 is not None and output1 is not None:
                        
                        input1 = input1.numpy().astype(np.bool)
                        output1 = output1.numpy().astype(np.bool)
                        self.grids_inputs.append(input1)
                        self.grids_outputs.append(output1)
                            
        self.grids_inputs = np.array(self.grids_inputs, dtype=np.bool)
        self.grids_outputs = np.array(self.grids_outputs, dtype=np.bool)

    def __len__(self):
        return len(self.grids_inputs)

    def __getitem__(self, item):
        
        if self.tensor_type == "float":
            return torch.from_numpy(self.grids_inputs[item]).float(), torch.from_numpy(self.grids_outputs[item]).float()
        
        if self.tensor_type == "long":
            return torch.from_numpy(self.grids_inputs[item]).long(), torch.from_numpy(self.grids_outputs[item]).long()

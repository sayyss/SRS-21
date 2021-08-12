import numpy as np
import json
import pandas as pd
import os
from copy import deepcopy
import multiprocessing as mp

all_ids = os.listdir('data/training/') + os.listdir('data/evaluation/')
id_maker = mp.Value('i', 2**28)

MAX_COLOR_PERMUTATIONS = 10    # number of color permutations can get up to 10! = 3.6e6


def shuffle_colors(pair):
    input_arr = np.array(pair['input'])
    output_arr = np.array(pair['output'])

    original = np.unique(np.append(input_arr, output_arr)).tolist()   # original color permutation
    color_structure = {
        'input': [],
        'output': []
    }

    for color in original:
        # get indexes that match
        color_structure['input'].append((input_arr == color).nonzero())
        color_structure['output'].append((output_arr == color).nonzero())

    shuffled = []

    for _ in range(MAX_COLOR_PERMUTATIONS):
        palette = np.random.choice(range(10), len(original), replace=False)
        new_input_arr = deepcopy(input_arr)
        new_output_arr = deepcopy(output_arr)

        # copy new colors into the new array, while preserving the color sturcture
        for idx, color in enumerate(palette):
            new_input_arr[color_structure['input'][idx]] = color
            new_output_arr[color_structure['output'][idx]] = color
        pair = {
            'input': new_input_arr.tolist(),
            'output': new_output_arr.tolist()
        }
        shuffled.append(pair)

    return shuffled


def rotate(arr):
    arr = np.array(arr)
    m, n = arr.shape
    new_arr = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        new_arr[i, :] = arr[:, n - i - 1]

    return new_arr.tolist()


def mirror(arr):
    arr = np.array(arr)
    m, n = arr.shape
    new_arr = np.zeros((m, n), dtype=np.int64)
    for i in range(n):
        new_arr[:, i] = arr[:, n - i - 1]

    return new_arr.tolist()


def make_id():
    global all_ids
    global id_maker
    id = all_ids[0]
    while id in all_ids:
        with id_maker.get_lock():
            id_maker.value += 1
            id = id_maker.value
        id = hex(id)[2:]
        id = str(id) + '.json'
    return id


def apply_transform(transform, data):
    new_data = deepcopy(data)
    for idx, pair in enumerate(data['train']):
        new_data['train'][idx]['input'] = transform(data['train'][idx]['input'])
        new_data['train'][idx]['output'] = transform(data['train'][idx]['output'])

    for idx, pair in enumerate(data['test']):
        new_data['test'][idx]['input'] = transform(data['test'][idx]['input'])
        new_data['test'][idx]['output'] = transform(data['test'][idx]['output'])

    return new_data


def apply_color_shuffle(id):
    data = json.load(open('data/training/' + id))

    new_train = [shuffle_colors(pair) for pair in data['train']]
    new_test = [shuffle_colors(pair) for pair in data['test']]

    new_ids = [id]

    for i in range(MAX_COLOR_PERMUTATIONS):
        new_data = deepcopy(data)
        for idx in range(len(data['train'])):
            new_data['train'][idx] = new_train[idx][i]

        for idx, pair in enumerate(data['test']):
            new_data['test'][idx] = new_test[idx][i]

        id = make_id()
        new_ids.append(id)
        json.dump(new_data, open('data/training/' + id, 'w'))


    return new_ids


def augment(data):
    new_ids = []
    augmneted_data = apply_transform(rotate, data)
    for _ in range(3):
        id = make_id()
        new_ids.append(id)
        json.dump(augmneted_data, open('data/training/' + id, 'w'))
        augmneted_data = apply_transform(rotate, augmneted_data)

    assert augmneted_data == data
    augmneted_data = apply_transform(mirror, data)
    for _ in range(4):
        id = make_id()
        new_ids.append(id)
        json.dump(augmneted_data, open('data/training/' + id, 'w'))
        augmneted_data = apply_transform(rotate, augmneted_data)

    return new_ids


def train_test_split():
    # 700 tasks for training, 100 + 100 for evaluation
    eval_ids = sorted(os.listdir('data/evaluation'))
    to_train = eval_ids[:300]
    tags = pd.read_csv('data/tags.csv', index_col=0)
    for id in to_train:
        os.rename('data/evaluation/' + id, 'data/training/' + id)
        tags.loc[id] = 0

    tags.to_csv('data/tags.csv')


def augment_data(num_color_shuffles=10):
    global MAX_COLOR_PERMUTATIONS
    MAX_COLOR_PERMUTATIONS = num_color_shuffles

    train_test_split()

    tags = pd.read_csv('data/tags.csv', index_col=0)

    train_ids = os.listdir('data/training/')
    np.random.seed(1)
    # apply rotations and symmetries
    for id in train_ids:
        data = json.load(open('data/training/' + id))
        augmented_ids = augment(data)
        added = pd.DataFrame([tags.loc[id]] * len(augmented_ids), index=augmented_ids)
        tags = tags.append(added)

    train_ids = os.listdir('data/training/')

    # apply color shuffles
    with mp.Pool(mp.cpu_count()) as p:
        colored_ids = p.map(apply_color_shuffle, train_ids)

    for ids in colored_ids:
        original_id = ids.pop(0)
        added = pd.DataFrame([tags.loc[original_id]] * len(ids), index=ids)
        tags = tags.append(added)

    tags.to_csv('data/tags.csv')


if __name__ == '__main__':
    augment_data()

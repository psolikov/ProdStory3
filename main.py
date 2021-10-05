import numpy as np

from copy import deepcopy
from math import floor

def read_data():
    data = []
    fname = "in.txt"
    f = open(fname, 'r')
    for i, line in enumerate(f):
        x, y = line.strip().split()
        data.append([float(x), float(y)])
    f.close()
    return np.array(data)

def write_data(data):
    fname = "out.txt"
    f = open(fname, 'w')
    f.write(f'{data[0]} {data[1]} {data[2]}')
    f.close()


def get_rank(data):
    data = np.array(sorted(data, key=lambda x: x[0]))
    data_copy = deepcopy(data)
    data_copy = [(data_copy[i],i) for i in range(len(data_copy))]
    data_copy, permutation = zip(*sorted(data_copy, key=lambda x: -x[0][1]))
    data_copy = np.array(data_copy)
    rank = np.empty(data_copy.shape[0])
    rank_all = np.arange(1, data_copy.shape[0] + 1)
    for i, d in enumerate(data_copy):
        rank[permutation[i]] = np.mean(rank_all[data_copy[:,1] == d[1]])
    return rank

def get_answers(data):
    if len(data) < 9:
        print('Error! Data should contain more than 9 pairs.')
        exit(1)
    rank = get_rank(data)
    N = data.shape[0]
    p = int(round(N/3))
    R1 = np.sum(rank[:p])
    R2 = np.sum(rank[-p:])
    err = int(round((N + 1/2) * np.sqrt(p / 6)))
    measure = round((R1 - R2) / (p * (N - p)), 2)
    return int(round(R1 - R2)), err, measure


if __name__ == "__main__":
    write_data([*get_answers(read_data())])
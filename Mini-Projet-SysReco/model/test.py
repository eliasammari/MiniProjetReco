

import sys

sys.path.append("..")
import numpy as np

import pandas as pd
from numpy import dot
from numpy.linalg import norm



def get_rowcsv(user_node, path):
        last = pd.read_csv(path)
        embedding = [last.loc[last['node'] == user_node].embedding]

        print(user_node)
       
        if  embedding[0].empty:
            return 0
        else: 

            embedding = [last.loc[last['node'] == user_node].embedding][0].values[0]
            chunks = embedding.split(' ')
            embd=  list(map(float, chunks)) # => [1,2,3]
            return embd

def get_sim_node2vec(u, k):
        
        embd_u = get_rowcsv(u,"embeddings/embeddings.csv")
        embd_v = get_rowcsv(k,"embeddings/embeddings.csv")
        if embd_u != 0 and embd_v != 0 :
            cos_sim = dot(embd_u, embd_v)/(norm(embd_u)*norm(embd_v))
            return cos_sim
        else:
             return 0


if __name__ == '__main__':
    print(get_sim_node2vec(1,384))

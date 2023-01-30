
import pandas as pd
from utility.similarity import pearson_sp, cosine_sp
from numpy import dot
from numpy.linalg import norm



def txt_to_csv(path):
    read_file = pd.read_csv(path)
    read_file.to_csv (r'embeddings/embeddings.csv')

def preprocess(path):
    txt_to_csv(path)
    df = pd.read_csv("embeddings/embeddings.csv")
    last = df.iloc[:,1].str.split(" ", 1, expand=True)  
    last = last.rename(columns={0: "node", 1: "embedding"})
    last[["node"]] = last[["node"]].apply(pd.to_numeric)
    last.to_csv (r'embeddings/embeddings.csv')
    


def get_row(user_node, path):
    last = pd.read_csv(path)
    embedding = [last.loc[last['node'] == user_node].embedding][0].values[0]
    chunks = embedding.split(' ')
    embd=  list(map(float, chunks)) # => [1,2,3]


    return embd




def get_sim(u, k):
    embd_u = get_row(u,"embeddings/embeddings.csv")
    embd_v = get_row(k,"embeddings/embeddings.csv")
    cos_sim = dot(embd_u, embd_v)/(norm(embd_u)*norm(embd_v))
    return cos_sim



if __name__ == '__main__':
    preprocess("embeddings/trust_data_embeddings_dim-128_p-1.0_q-1.0.txt")
    

    



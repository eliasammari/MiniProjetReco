# -*- coding: utf-8 -*-
"""Untitled18.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1D5jf1X0eKhgj9fzLurq1I4ZwCZtHe80o
"""

import matplotlib
import networkx as nx
import urllib
import csv

g = nx.read_edgelist("trust_data.txt",create_using=nx.DiGraph(), nodetype = int)

# check if the data has been read properly or not.


# count the number of nodes

print(g.number_of_nodes())

# number of self-nodes

def get_graph_info(graph):
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    #print("Available nodes:", list(graph.nodes))
    #print("Available edges:", list(graph.edges))
    if type(graph) == nx.classes.digraph.DiGraph:
        print("Connected components:", 
              list(nx.weakly_connected_components(graph)))
    else:
        print("Connected components:", list(nx.connected_components(graph)))
    print("Node degree:", dict(graph.degree()))

get_graph_info(g)

import pandas as pd

# get the local cluster coefficient dataframe
def get_local_cluster_coefficient(graph):
    LCC_df = pd.DataFrame(sorted(nx.clustering(graph).items(), 
                                 key=lambda item: -item[1]), 
                      columns=["node", "LCC"])
    return LCC_df

# find nodes with LCC score of > 0.5:
# means that > 50% of change a node n's friends are friends of each other
node_data = dict(g.nodes())
LCC_df = get_local_cluster_coefficient(g)

print(round(100 * len(LCC_df.query("LCC > 0.5")) / len(LCC_df), 1),
      "% of the nodes have Local Cluster Coefficients > 0.5")

# 2.2.2 Average local clustering coefficient
import numpy as np
np.mean(LCC_df["LCC"])

nx.average_clustering(g)

# Transitivity:
nx.transitivity(g)

largest = g.subgraph((max(nx.strongly_connected_components(g), key=len)))

get_graph_info(largest)

# Average Distance (Average Shortest Path Length)
import math
math.ceil(nx.average_shortest_path_length(largest))

# get the local cluster coefficient dataframe
def get_eccentricity_df(graph):
    eccentricity_df = pd.DataFrame(sorted(nx.eccentricity(graph).items(), 
                                 key=lambda item: -item[1]), 
                      columns=["node", "eccentricity"])
    return eccentricity_df

# get sorted accentricty df
eccentricity_df = get_eccentricity_df(largest)
eccentricity_df["eccentricity"].plot.hist(figsize=(8,5), title="Eccentricity Distribution");

nx.radius(largest)

nx.diameter(largest)

nx.center(largest)

nx.periphery(largest)

nx.density(g)

nx.node_connectivity(g)

nx.minimum_node_cut(g)

in_degree_centrality_df = pd.DataFrame(nx.in_degree_centrality(g).items(), 
                                       columns=["node", "in_degree_centrality"])
in_degree_centrality_df.sort_values("in_degree_centrality", ascending=False).head(10)

out_degree_centrality_df = pd.DataFrame(nx.out_degree_centrality(g).items(), 
                                       columns=["node", "out_degree_centrality"])
out_degree_centrality_df.sort_values("out_degree_centrality", ascending=False).head(10)

closeness_centrality_df = pd.DataFrame(nx.closeness_centrality(g).items(), 
                                       columns=["node", "closeness_centrality"])
closeness_centrality_df.sort_values("closeness_centrality", ascending=False).head(10)

betweenness_centrality_df = pd.DataFrame(nx.betweenness_centrality(g).items(), 
                                       columns=["node", "betweenness_centrality"])
betweenness_centrality_df.sort_values("betweenness_centrality", ascending=False).head(10)

edge_betweenness_centrality_df = pd.DataFrame(nx.edge_betweenness_centrality(g).items(), 
                                       columns=["edge", "betweenness_centrality"])
edge_betweenness_centrality_df.sort_values("betweenness_centrality", ascending=False).head(10)

pagerank_centrality_df = pd.DataFrame(nx.pagerank(g).items(), 
                                       columns=["node", "pagerank_centrality"])
pagerank_centrality_df.sort_values("pagerank_centrality", ascending=False).head(10)

# compute hub and auth centrality scores
hub_scores, auth_scores = nx.hits(g)
hub_centrality_df = pd.DataFrame(hub_scores.items(), 
                           columns=["node", "hub_centrality"])
auth_centrality_df = pd.DataFrame(auth_scores.items(), 
                           columns=["node", "auth_centrality"])
hub_centrality_df.sort_values("hub_centrality", ascending=False).head(10)

auth_centrality_df.sort_values("auth_centrality", ascending=False).head(10)

# combine centrality measurement results
centrality_summary = in_degree_centrality_df\
    .merge(out_degree_centrality_df, on="node")\
    .merge(pagerank_centrality_df, on="node")\
    .merge(closeness_centrality_df, on="node")\
    .merge(betweenness_centrality_df, on="node")\
    .merge(hub_centrality_df, on="node")\
    .merge(auth_centrality_df, on="node")
centrality_summary


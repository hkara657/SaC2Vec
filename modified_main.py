
# coding: utf-8

# In[ ]:

'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import modified_node2vec
from gensim.models import Word2Vec


# In[ ]:

def simulate_walks(self, num_walks, walk_length):
    '''
    Repeatedly simulate random walks from each node.
    '''
    G = self.G[0]  #we can take any graph as we just need to find the nodes
    walks = []
    nodes = list(G.nodes())
    print 'Walk iteration:'
    for walk_iter in range(num_walks):
        self.ct = [0,0]
        print str(walk_iter+1), '/', str(num_walks)
        random.shuffle(nodes)
        for node in nodes:
            walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        print self.ct
    return walks
		


# In[ ]:

def simulate_walks(self, num_walks, walk_length):
    '''
    Repeatedly simulate random walks from each node.
    '''
    G = self.G[0]  #we can take any graph as we just need to find the nodes
    walks = []
    nodes = list(G.nodes())
    print 'Walk iteration:'
    for walk_iter in range(num_walks):
        self.ct = [0,0]
        print str(walk_iter+1), '/', str(num_walks)
        random.shuffle(nodes)
        for node in nodes:
            walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        print self.ct
    return walks



# In[ ]:

def read_graph():
	'''
	Reads the input network in networkx.
	'''
		#C will always be weighted
	C = nx.read_edgelist(input_C, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	
	if weighted_R:
		R = nx.read_edgelist(input_R, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		R = nx.read_edgelist(input_R, nodetype=int, create_using=nx.DiGraph())
		for edge in R.edges():
			R[edge[0]][edge[1]]['weight'] = 1

	if not directed_R:
		R = R.to_undirected()
		
	if not directed_C:
		C = C.to_undirected()
	return R,C

def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, workers=workers, iter=1)
    model.wv.save_word2vec_format(output)
#     model.save_word2vec_format(output)
    return


# In[ ]:

for dataset in ['MSA']:
    
    path = '../../data/'+dataset
    print path
    
    input_R = path+'/reference.edgelist'
    # input_R = 'test_r'
    input_C =path+'/cosine_cont.edgelist'
    # input_C = 'test_c'
    output =  path+'/two_level.embed'
    directed_C = True     # C is always weighted
    directed_R = False    #undirecetd
    weighted_R = False
    p_R = p_C = q_R = q_C = 1
    workers = 8
    iter = 1
    window_size = 10
    walk_length = 80
    dimensions = 128
    num_walks = 10


    print '\n reading graph'
    nx_R, nx_C = read_graph()

    print '\n creating object'
    G = modified_node2vec.Graph(nx_R, nx_C ,directed_R, directed_C, p_R, q_R, p_C, q_C)


    #### we can save the object G
    print G.G[0].number_of_nodes()
    print G.G[1].number_of_nodes()

    print '\nlevel trans 1 started'
    G.get_level_transition_weight(0)     # prob for changing the levels
    print '\nlevel trans 2 started'
    G.get_level_transition_weight(1)


    print '\nalias 1 started'
    G.preprocess_transition_probs(0)
    print '\nalias 2 started'
    G.preprocess_transition_probs(1)


    print '\nwalk started'
    walks = G.simulate_walks(num_walks, walk_length)
    learn_embeddings(walks)
    print 'finally count of ',G.ct
#     f1.conv_to_csv( dataset, 'two_level' )


# In[ ]:

# output =  path+'/restart.embed'
# learn_embeddings(walks)
# print 'finally count of ',G.ct
# f1.conv_to_csv( dataset, 'restart' )


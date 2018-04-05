import numpy as np
import networkx as nx
import random


class Graph():
	def __init__(self, nx_R, nx_C, is_directed1, is_directed2, p1, q1, p2, q2):
		self.G = [nx_R, nx_C]
		self.is_directed = [is_directed1, is_directed2]
		self.p = [p1,p2]
		self.q = [q1,q2]
		self.alias_nodes = [None, None]
		self.alias_edges = [None, None]
		self.trans_weight = [None,None]
		self.ct =  [0,0]

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		#~ print 'length is ',len(self.trans_weight[0]),len(self.trans_weight[1])
		#~ print 'nodes ',self.G[0].number_of_nodes(),self.G[1].number_of_nodes()
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			
			#now we need to see if we need to change the graph or not
			###  i.e. to use R or C
			### taking upper as reference and lower level as content
			up = self.trans_weight[1][cur]   # weight to go in Reference
			down = self.trans_weight[0][cur]   # weight to go in Content
			pu = up/(up+down)   
			pd = 1-pu
			
			x = random.random()  # random num between 0---1
			if x<pu:    # if pu is large then more chances of Reference being selected
			    ind = 0
			else:
			    ind = 1
			
			self.ct[ind] += 1   #to get count in which graph the Random walk is in.
			
			cur_nbrs = sorted(self.G[ind].neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[ind][cur][0], alias_nodes[ind][cur][1])])
				else:
					prev = walk[-2]
					if (prev, cur) not in alias_edges[ind]:   #when the edge is not in other graph
						walk.append(cur_nbrs[alias_draw(alias_nodes[ind][cur][0], alias_nodes[ind][cur][1])])
					else:
						e1 = alias_edges[ind][(prev, cur)][0]
						e2 = alias_edges[ind][(prev, cur)][1]
						
						pr = (e1,e2)
						tmp = alias_draw(e1,e2)
						next = cur_nbrs[tmp]
						walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G[0]  #we can take any graph as we just need to find the nodes
		print 'new 1'
		walks = []
		nodes = list(G.nodes())
		print 'Walk iteration:'
		for walk_iter in range(num_walks):
			print str(walk_iter+1), '/', str(num_walks)
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
				
			print self.ct
			self.ct = [0,0]
		return walks
		
	def get_level_transition_weight(self, ind):
		####
		G = self.G[ind]
		
		mat = nx.to_scipy_sparse_matrix(G)
		
		if ind==0:
		    avg=1.0
		else:
		    avg = 1.0*np.sum(mat)/G.number_of_edges()

		print 'thr is 1*avg ', 1*avg
		mat = mat>=avg
		tau = np.sum(mat,axis=1)
		self.trans_weight[ind] = np.log(np.e  + tau)
		###
		
		#~ G = self.G[ind]
		#~ avg = 0.0
		#~ print G.number_of_nodes()
		#~ for e in G.edges():
			#~ avg += G[e[0]][e[1]]['weight']
		#~ avg /= len(G.edges())
		#~ trans_weight = np.zeros(G.number_of_nodes())
		
		#~ for src in G.nodes():
			#~ ct=0
			#~ for dest in G.neighbors(src):   #outgoing edges
				#~ if G[src][dest]['weight']>avg:
					#~ ct+=1
			#~ trans_weight[src] = np.log(np.e + ct)
		#~ self.trans_weight[ind] = trans_weight


	def get_alias_edge(self, src, dst, ind):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G[ind]
		p = self.p[ind]
		q = self.q[ind]

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self,ind):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G[ind]
		print 'kkk'
		is_directed = self.is_directed[ind]

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1], ind)
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1], ind)
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0], ind)

		self.alias_nodes[ind] = alias_nodes
		self.alias_edges[ind] = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)
	kk = int(np.floor(np.random.rand()*K))
	tmp = q[kk]
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]

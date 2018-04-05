
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import os
import networkx as nx


# In[ ]:

dataset = 'cora'
path = '../../data/'+dataset

print path


# #### not possible to read file directly using pandas
# #### so first read then split 

# In[ ]:

if dataset in ['MSA','Wiki']:
    f=open(path+'/content.csv','r')
    data = f.read()
    f.close()

    data = data.split('\n')

    if len(data[-1])==0:
        data.pop()

    print len(data)

    tmp = []
    for i in np.arange(len(data)):
        tmp.append(data[i].split(' '))

    cont = np.array(tmp)
    cont = cont.astype('float')
    print cont.shape
    del data
    del tmp
else:
    cont = pd.read_csv(path+'/content.csv',header=None)
    print cont.shape


# In[ ]:

#cosine similarity
cw = np.matmul(cont,cont.T)
if dataset=='MSA':
    cw[13988][:] = np.ones(cw.shape[1])/cw.shape[1]

norm = np.linalg.norm(cont,axis=1)
if dataset=='MSA':
    norm[13988] = 1
norm = np.reshape(norm,(len(norm),1))
norm_mat = np.matmul(norm, norm.T)

cw = cw/norm_mat
cw.shape


# In[ ]:

#making the diagonal entries as 0
n = cont.shape[0]
ind = np.diag_indices(n)
cw[ind]=0


# In[ ]:

tmp = (np.sum(cw,axis=1)==0)

#if all are zero in a row then make all outgoing edges same
for i in np.arange(cw.shape[0]):
    if tmp[i]==True:
        print i
        cw[i][:] = np.ones(cw.shape[1])/cw.shape[1]
        
# 13988 for MSA


# In[ ]:

count=0


# In[ ]:

edge_dict = {'cora':7,'citeseer':7,'pubmed':3,'MSA':50,'Wiki':45}
num_edges = edge_dict[dataset]


# In[ ]:

print path
print num_edges


# In[ ]:

f=open(path+'/cosine_cont.edgelist','w')
for i in np.arange(cw.shape[0]):
    row = -cw[i,:]  # to get in decsending order
    ind = np.argsort(row)   #get indices
    if cw[i][ind[0]]!=0:
        count+=1
#     for j in np.arange(int(edge_percent*cw.shape[1])):   #get top 40% indices
    for j in np.arange(num_edges):   ###only top 100
        
        if cw[i][ind[j]]==0:  #bcz. if it is 0 then after this all will be zero only
            break
        f.write(str(i)+' '+str(ind[j])+' '+str(cw[i][ind[j]])+'\n')
f.close()


# In[ ]:

R = nx.read_edgelist(path+'/reference.edgelist', nodetype=int, create_using=nx.DiGraph())

#since unweighted
for edge in R.edges():
    R[edge[0]][edge[1]]['weight'] = 1
    
# since undirected
R = R.to_undirected()

R = np.array(nx.to_numpy_matrix(R))
R.shape


# In[ ]:

comb = R + cw
print comb.shape
# comb = np.array(comb)
# print comb.shape


# In[ ]:

f=open(path+'/graph_sum.edgelist','w')
for i in np.arange(comb.shape[0]):
    row = -comb[i,:]  # to get in decsending order
    ind = np.argsort(row)   #get indices
    if comb[i][ind[0]]!=0:
        count+=1
#     for j in np.arange(int(edge_percent*comb.shape[1])):   #get top 40% indices
    for j in np.arange(num_edges):   ###only top 100
        
        if comb[i][ind[j]]==0:  #bcz. if it is 0 then after this all will be zero only
            break
        f.write(str(i)+' '+str(ind[j])+' '+str(comb[i][ind[j]])+'\n')
f.close()


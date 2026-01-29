import numpy as np


def edgelist_to_adjmat(edgelist):
    '''Turn an edgelist into an adjacency matrix
    
    edgelist is a numpy array of source-sink pairs. 

    indices is the order in which the columns/rows are ordered. 
    '''

    sources, sinks = edgelist.T

    all_ids = set(sources) | set(sinks)

    n_all_ids = len(all_ids)

    adj_mat = np.zeros((n_all_ids,n_all_ids))

    i = 0
    ind_dict = {}
    ind_to_id = []
    for edge in edgelist:
        
        source_edge = ind_dict.get(edge[0])
        if source_edge==None:
            ind_dict[edge[0]] = i
            source_edge = i
            ind_to_id.append(edge[0])
            i+=1
        sink_edge = ind_dict.get(edge[1])
        if sink_edge==None:
            ind_dict[edge[1]] = i
            sink_edge = i
            ind_to_id.append(edge[1])

            i+=1

        adj_mat[source_edge, sink_edge] = 1

    return adj_mat, ind_dict, ind_to_id
        
def edgelist_to_bptadjmat(edgelist):
    '''Turn an edgelist into an bipartite adjacency matrix

    First column and second column are treated independently in the adjacency matrix
    DOESN't necessarily produce a square matrix.
    
    edgelist is a numpy array of source-sink pairs. 

    indices is the order in which the columns/rows are ordered. 
    '''

    sources, sinks = edgelist.T

    all_ids = set(sources) | set(sinks)

    adj_mat = np.zeros((len(set(sources)),len(set(sinks)))) # sources/sinks on first/second axis

    so_i = 0
    si_i = 0

    source_ind_dict = {}
    sink_ind_dict = {}

    source_ind_to_id = []
    sink_ind_to_id = []

    for edge in edgelist:
        
        source_edge = source_ind_dict.get(edge[0])
        if source_edge==None:
            source_ind_dict[edge[0]] = so_i
            source_edge = so_i
            source_ind_to_id.append(edge[0])
            so_i+=1
        sink_edge = sink_ind_dict.get(edge[1])
        if sink_edge==None:
            sink_ind_dict[edge[1]] = si_i
            sink_edge = si_i
            sink_ind_to_id.append(edge[1])

            si_i+=1

        adj_mat[source_edge, sink_edge] = int(1)

    return adj_mat, source_ind_to_id, sink_ind_to_id

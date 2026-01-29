import numpy as np


def dict_to_sim_mat(node_types_1, node_types_2, sim_val=1):
    '''Dicts are indexed by the indices of matrix 1 and 2 respectively 
    the similarity is given by matching the node types via the dictionary. 
    
    The dictionary is indexed by integers that correspond to indices of the adjacency matrix to match. 
    '''

    S = np.zeros((len(node_types_1), len(node_types_2)))

    for i in range(len(node_types_1)):
        for j in range(len(node_types_2)):
            if node_types_1[i] == node_types_2[j]:
                S[i,j] = sim_val
    return S

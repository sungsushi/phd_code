'''
This is a fast vectorisation package for (un)directed, (un)weighted and bipartite graphs. 
by Sung Soo Moon
'''

import numpy as np
import scipy as sp 
import pandas as pd



def bipartite_cooarray(df, row_col, row_order=None, col_order=None, weight=True):
    """
    
    Turns a bipartite edge list df into a sparse COOrdinate array. 

    Parameters
    ----------
    df : pandas.DataFrame
        The bipartite graph edge list with columns ``row_col`` and ``'weight'`` if ``weight=True``.

    row_col : {list, array} of strings of length 2
        ``row_col[0]`` and ``row_col[1]`` are strings specifying the row and column name 
        in the dataframe, respectively. 

    row_order : {list, array}, default=``None``
        Specifies the order in which row nodes appear. Not required to specify all the
        elements e.g. an input of ``['node_1'] ``will put ``'node_1'`` first, then will enumerate
        the rest of the elements as they appear. If ``None``, then will enumerate elements in 
        order of appearance in the dataframe. 

    col_order : {list, array}, default=``None``
        Specifies the order in which column nodes appear. Not required to specify all the
        elements, also see ``row_order``. If ``None``, then will enumerate elements in 
        order of appearance in the dataframe. 

    weight : bool, default=``True``
        Indexes the dataframe for ``'weight'`` column for appropriate weights to initialise the 
        matrix. If ``False``, then intialises with all weights as 1. 


    Returns 
    -------
    coo : scipy.sparse.coo_array of shape (a, b)
        COOrdinate sparse array of the bipartite edge list ``df``. E.g. when ``weights=False``, 
        entry (i, j) is 1 if there's an edge between i (from ``df[row_col[0]]``) and j 
        (from ``df[row_col[1]]``) in ``df``.

    row_unique : array of shape (a, )
        Unique elements in ``df[row_col[0]]`` appearing in the order of appearence or specified
        by ``row_order``.

    col_unique : array of shape (b, )
        Unique elements in ``df[row_col[1]]`` appearing in the order of appearence or specified
        by ``col_order``.


    Notes
    -----

    Part of the fast vectorisation routine. 

    """
    row, col = row_col

    row_to_factorize = list(df[row])
    col_to_factorize = list(df[col])
    r_offset = 0
    c_offset = 0

    if row_order is not None:
        row_to_factorize = list(row_order) + row_to_factorize
        r_offset = len(row_order)
    if col_order is not None:
        col_to_factorize = list(col_order) + col_to_factorize
        c_offset = len(col_order)

    proto_row, row_unique = pd.factorize(row_to_factorize)
    proto_col, col_unique = pd.factorize(col_to_factorize)

    row_coord = proto_row[r_offset:]
    col_coord = proto_col[c_offset:]

    if weight==True:
        w_data= df['weight']
    else:
        w_data=np.ones(len(col_coord))

    coo = sp.sparse.coo_array((w_data, (row_coord, col_coord)), shape=(len(row_unique), len(col_unique)))
    return coo, row_unique, col_unique


def adjacency_cooarray(df, row_col, id_order=None, weight=True, directed=True):
    """
    
    Turns an edge list df into a square sparse COOrdinate array. 

    Parameters
    ----------
    df : ``pandas.DataFrame``
        The graph edge list with columns row_col and 'weight' if weight=True.

    row_col : {list, array} of strings of length 2
        ``row_col[0]`` and ``row_col[1]`` are strings specifying the row and column name 
        in the dataframe, respectively. 

    id_order : {list, array}, default=``None``
        Specifies the order in which rows and column nodes appear. Not required to specify all the
        elements e.g. an input of ``['node_1'] ``will put ``'node_1'`` first, then will enumerate
        the rest of the elements as they appear. If ``None``, then will enumerate elements in 
        order of appearance in the dataframe. 

    weight : bool, default=``True``
        Indexes the dataframe for ``'weight'`` column for appropriate weights to initialise the 
        matrix. If ``False``, then intialises with all weights as 1. 

    directed : bool, default=``True``
        If the edge list is a directed list, then the resulting sparse array will reflect this, 
        where ``row_col[0]`` is taken to be the pre-edge node and the ``row_col[1]`` is the post-edge 
        node. If ``False`` then the resulting sparse array will be a symmetric array. 

    Returns 
    -------
    coo : scipy.sparse.coo_array of shape (a, a)
        COOrdinate sparse array of the bipartite edge list ``df``. E.g. when ``weights=False``, 
        entry ``(i, j)`` is 1 if there's an edge between i (from ``df[row_col[0]]``) and j 
        (from ``df[row_col[1]]``) in ``df``.

    node_names : array of shape (a, )
        Unique nodes names in ``df[row_col]`` appearing in the order of appearence or specified
        by ``id_order``.

    Notes
    -----

    Part of the fast vectorisation routine. 


    """

    edges = df[row_col].values

    if id_order is None:
        id_order=[]

    flat_nodes = pd.Series(list(id_order) + list(edges.flatten())) # get enumerated codes
    proto_codes, node_names = pd.factorize(flat_nodes)
    codes = proto_codes[len(id_order):]
    codes = codes.reshape(-1, 2)
    row_coord, col_coord = codes[:, 0], codes[:, 1]

    if weight==True:
        w_data= df['weight']
    elif weight==False:
        w_data=np.ones(len(codes))

    if directed:
        coo = sp.sparse.coo_array((w_data, (row_coord, col_coord)), shape=(len(id_order), len(id_order)))
    if not directed: # duplicate the entries for symmetric undirected graph
        row_coord = list(row_coord)
        col_coord = list(col_coord)
        w_data = list(w_data)
        coo = sp.sparse.coo_array((w_data + w_data, (row_coord + col_coord, col_coord + row_coord)), shape=(len(id_order), len(id_order)))

    return coo, node_names

def csr_row_norm(csrarr):
    """ 
    
    Divides the compressed array by its row sum, except for rows that sum to 0, which are preserved.

    Matrix should ONLY have positive values...

    Parameters
    ----------
    csrarr : ``scipy.sparce.csr_array``
        A csr array we wish to normalise the rows by dividing each element by its row sum. 
        We ignore rows that sum to zero (i.e. a row of zeros)

    Returns 
    -------
    csrarr_normalised : ``scipy.sparse.csr_array ``
        The normalised csr array.

    Notes
    -----

    Part of the fast vectorisation routine. 

    """

    row_sums = np.array(csrarr.sum(axis=1)).flatten()  
    row_sums[row_sums == 0] = 1 # divide by 1 instead of zero
    inv_row_sums = sp.sparse.diags(1 / row_sums)
    csrarr_normalised = inv_row_sums @ csrarr
    return csrarr_normalised


def csr_col_norm(csrarr):
    """ 
    
    Divides the compressed array by its column sum, except for columns that sum to 0, which are preserved.

    Matrix should ONLY have positive values...

    Parameters
    ----------
    csrarr : ``scipy.sparce.csr_array``
        A csr array we wish to normalise the columns by dividing each element by its column sum. 
        We ignore columns that sum to zero (i.e. a row of zeros)

    Returns 
    -------
    csrarr_normalised : ``scipy.sparse.csr_array`` 
        The normalised csr array.

    Notes
    -----

    Part of the fast vectorisation routine. 

    """    
    col_sums = np.array(csrarr.sum(axis=0)).flatten()  
    col_sums[col_sums == 0] = 1 # divide by 1 instead of zero
    inv_col_sums = sp.sparse.diags(1 / col_sums)
    csrarr_normalised = csrarr @ inv_col_sums 
    return csrarr_normalised


def vectorisation(edge_df, meta_df, edge_row_cols, meta_row_cols, \
                  edge_weight=True, meta_weight=True, id_order=None, \
                  redundant_remove=False):
    """ 
    
    Creates data frame of vectors by their node ids as indices and metadata annotations
    as the columns. For DIRECTED graphs. 

    Parameters
    ----------
    edge_df : ``pandas.DataFrame``
        The directed graph edge list with columns in edge_row_col,
        and ``'weight'`` if ``edge_weight=True``.

    meta_df : ``pandas.DataFrame``
        The bipartite metadata graph edge list with columns meta_row_col, 
        and ``'weight'`` if ``meta_weight=True``.

    edge_row_cols : {list, array}
        List of strings specifying the row and column names in ``edge_df``. 
        By convention, we take the row to be the pre-edge, and column to be the post-edge, 
        i.e. each element of ``edge_row_cols[0]`` makes directed edges to each
        element of ``edge_row_cols[1]``.

    meta_row_cols : {list, array}
        List of strings specifying the nodes and annotation column names in ``meta_edge_df``. 
        The ``meta_row_cols[0]`` should be consistent with the nodes in ``edge_df``, and 
        ``meta_row_cols[1]`` should be the annotation metadata for each node.

    edge_weight : bool, default=``True``
        ``True`` if ``edge_df`` has a ``'weight'`` column to use to weight the 
        connectivity aggregation. If ``False``, then weight all edge equally 
        (weights of 1). 

    meta_weight : bool, default=True
        ``True`` if ``meta_df`` has a ``'weight'`` column to use to weight the 
        annotation aggregation. If ``False``, then weight all annotation edges equally 
        (weights of 1). 

    id_order : {list, array}, default=``None``
        Specifies the order in which indices of the vector dataframe appear. Not required 
        to specify all the elements e.g. an input of ``['node_1'] ``will put ``'node_1'`` first,
        then will enumerate the rest of the elements as they appear. If ``None``, then will 
        enumerate elements in order of appearance in the dataframe. 

    redundant_remove : bool, default=``False``
        If ``True``, remove empty rows or columns in the final output (entries that are all zeros)

    Returns 
    -------
    all_norm_vec_df : ``pandas.DataFrame``
        Normalised vector dataframe. 

    Notes
    -----

    Part of the fast vectorisation routine. 

    """    
    adj_AA_coo, _id_order = adjacency_cooarray(df=edge_df, 
                                               row_col=edge_row_cols, 
                                               id_order=id_order, 
                                               weight=edge_weight, 
                                               directed=True) 
    
    bpt_AB_coo, row_order, col_order = bipartite_cooarray(df=meta_df, 
                                                          row_col=meta_row_cols, 
                                                          weight=meta_weight, 
                                                          row_order=_id_order)

    adj_AA_out = (adj_AA_coo @ bpt_AB_coo).tocsr() # out matrix
    adj_AA_in = (adj_AA_coo.T @ bpt_AB_coo).tocsr() # in matrix

    # out matrix normalisation:
    adj_AA_out_normalised = csr_row_norm(adj_AA_out)

    # in matrix normalisation:
    adj_AA_in_normalised = csr_row_norm(adj_AA_in)

    # # put together into one dataframe:
    # all_col_names = np.concatenate([col_order + '_out', col_order + '_in'])
    # all_norm_vec_df = pd.DataFrame(sp.sparse.hstack([adj_AA_out_normalised, adj_AA_in_normalised]).toarray(), \
    #                             columns=all_col_names, index=row_order)

    # by default, keep all:
    out_col_keep, in_col_keep = [[True for _ in col_order] for i in range(2)]
    row_keep = [True for _ in row_order]

    if redundant_remove:
        # remove: redundant columns and rows that are empty
        out_col_sums = np.array(adj_AA_out_normalised.sum(axis=0)).flatten()  
        out_col_keep = (out_col_sums != 0)
        in_col_sums = np.array(adj_AA_in_normalised.sum(axis=0)).flatten()  
        in_col_keep = (in_col_sums != 0)

        out_row_sums = np.array(adj_AA_out_normalised.sum(axis=1)).flatten()  
        in_row_sums = np.array(adj_AA_in_normalised.sum(axis=1)).flatten()  
        out_row_keep = (out_row_sums != 0)
        in_row_keep = (in_row_sums != 0)
        row_keep = out_row_keep + in_row_keep # only false if both in/out are false...

    # columns and rows to keep:
    adj_AA_out_normalised = adj_AA_out_normalised[:, out_col_keep][row_keep,:]
    adj_AA_in_normalised = adj_AA_in_normalised[:, in_col_keep][row_keep,:]

    # column and row names to keep:
    all_col_names = np.concatenate([col_order[out_col_keep] + '_out', 
                                    col_order[in_col_keep] + '_in'])
    all_row_names = np.array(row_order)[row_keep]

    # construct a dataframe:
    all_norm_vec_df = pd.DataFrame(sp.sparse.hstack([adj_AA_out_normalised, adj_AA_in_normalised]).toarray(), \
                                   columns=all_col_names, index=all_row_names)
    return all_norm_vec_df
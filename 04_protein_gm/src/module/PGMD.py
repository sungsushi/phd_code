import numpy as np
import pandas as pd
from .gmd import distance_stats
import networkx as nx
from graspologic.match import graph_match


def get_orientation_multiadj(adj, orientation, types=False):
    ''' Nonbinary version: '''
    # adj = (adj * orientation).fillna(0).astype(int)

    if not types: 
        types = list(set(orientation.replace([np.inf], 0).astype(int).to_numpy().flatten()) - {0.})
    # print(types)

    multiadj = np.zeros((len(types), len(adj), len(adj)))
    # print(multiadj.shape)
    for t in range(len(types)):
        # print(types[t])
        t_mat = (orientation==types[t])*adj
        multiadj[t, :len(adj), :len(adj)] = t_mat

    return multiadj

def get_orientation_2(fpath):
    '''output the orientation array'''
    # fpath= f'./data/secondary_structure_contact_map/orientation/{p_id}.txt'
    orientation = pd.read_csv(fpath, delimiter = "\t")
    orientation = orientation.iloc[:, :-1]
    orientation.set_index('ss_ele', inplace=True)
    orientation.replace(to_replace='-', value=np.inf, inplace=True)
    orientation = orientation.apply(pd.to_numeric)
    return orientation

def logistic(x):
    return 1/(1+np.exp(-x))

def length_similiarity(lengths, alpha, gamma):
    '''length similarity function'''
    delta = abs(lengths[0]-lengths[1])
    g_x = alpha * (1 - logistic(gamma*(delta-7.5)))
    return g_x

def invd_sim_func(adj):
    return abs(1/(1+adj))

def forward_chain_adj_prep(val):
    '''Input the value of the similarity to create an adjacency matrix where there are edges between all consecutive elements.'''
    def _chain_adj(mat):
        size = len(mat)
        chain = np.zeros((size, size))
        for i in range(size-1):
            chain[i, i+1] = val
            # chain_adj[i+1, i] = val
        return chain
    return _chain_adj

class SSE_read:
    def __init__(self, fpathprefix, opathprefix, sim_func, chain_func=None, chainval=0, fpathext='ssnw', opathext='txt', threshold=None):
        self.fpathprefix = fpathprefix
        self.opathprefix = opathprefix
        self.fpathext = fpathext
        self.opathext = opathext
        self.fpath = None
        self.opath = None
        self.cmat = None
        self.binarised_cmat = None
        self.orientation = None
        self.adj = None
        self.sim_func = sim_func
        self.chain_func = chain_func
        if threshold==None:
            self.threshold = 7.5
        else: 
            self.threshold = threshold
        self.chainval = chainval
        param_fields = ['raw', 'processed', 'omulti', 'chain']
        self.params = {i:False for i in param_fields}
        self.nxgraph = None
 
    def set_pid(self, pid):
        self.pid = pid
        self.fpath = self.fpathprefix + '/' + f'{self.pid}.{self.fpathext}'
        self.opath = self.opathprefix + '/' + f'{self.pid}.{self.opathext}'
        return self

    def get_contact_matrix(self):
        # fpath = self.fpathprefix + '/'+ f'{self.pid}.ssnw'
        '''prepares the distance matrix of the contact map'''
        cmat = pd.read_csv(self.fpath, delimiter = "\t")
        cmat = cmat.iloc[:, :-1]
        cmat.set_index('ss_ele', inplace=True)
        cmat.replace(to_replace='-', value=np.inf, inplace=True)
        cmat = cmat.apply(pd.to_numeric)
        return cmat

    # for old ssnw false positives code:
    # def __get_contact_matrix(self):
    #     # fpath = self.fpathprefix + '/'+ f'{self.pid}.ssnw'
    #     data = pd.read_csv(self.fpath, delimiter = "\t")
    #     cmat = data.iloc[1:, 2:].dropna(axis=0, how='all').dropna(axis=1, how='all')
    #     lengths = data.Length[cmat.index.values].to_numpy()
    #     ele_inds = data.iloc[:,1][cmat.index.values].to_numpy()
    #     cmat.reset_index(drop=True, inplace=True)
    #     cmat.columns = cmat.index.values    
    #     new_indices = []
    #     for ele, length in zip(ele_inds, lengths):
    #         new_indices.append(ele[0] + str(int(length)))
    #     cmat.insert(loc=0, column='ss_ele', value=new_indices)
    #     cmat.replace(to_replace='-', value=np.inf, inplace=True)
    #     full_lengths = data.Length[1:].to_list()
    #     full_ssetypes = data.iloc[:,1][1:].to_list()
    #     full_sses = []
    #     for ele, length in zip(full_ssetypes, full_lengths):
    #         full_sses.append(ele[0] + str(int(length)))
    #     cmat.set_index('ss_ele', inplace=True)
    #     cmat = cmat.apply(pd.to_numeric)
    #     return cmat
    
    def __get_contact_matrix(self):
        # fpath = self.fpathprefix + '/'+ f'{self.pid}.ssnw'
        '''prepares the distance matrix of the contact map'''
        cmat = pd.read_csv(self.fpath, delimiter = "\t")
        cmat = cmat.iloc[:, :-1]
        cmat.set_index('ss_ele', inplace=True)
        cmat.replace(to_replace='-', value=np.inf, inplace=True)
        cmat = cmat.apply(pd.to_numeric)

        return cmat


    
    def __get_orientation_matrix(self, correction=False):
        # opath = self.opathprefix + '/' + f'{self.pid}.txt'
        orientation = pd.read_csv(self.opath, delimiter = "\t")
        orientation = orientation.iloc[:, :-1]
        orientation.set_index('ss_ele', inplace=True)
        orientation.replace(to_replace='-', value=np.inf, inplace=True)
        orientation = orientation.apply(pd.to_numeric)
        # values 2 and 3 refer to orthogonal in the orientation matrix. 
        if correction:
            orientation = orientation.replace(2, 3) 
        return orientation
    
    def reset_SSE(self):
        # TODO: nicer way to reset params...
        self.params['raw'] = False
        self.cmat = None
        self.orientation = None
        self.nxgraph = None
        self.binarised_cmat = None
        self.adj = None
        param_fields = ['raw', 'processed', 'omulti', 'chain']
        self.params = {i:False for i in param_fields}
        self.fpath = None
        self.opath = None


    def __prep_graph_obj(self, threshold):
        # if threshold==None:
        # threshold = self.threshold
        pid = self.pid 
        cmat = self.cmat
        thresholded = self.__binarise(mat=cmat, threshold=threshold)
        node_names = thresholded.index.tolist()
        cmat_array = thresholded.to_numpy()
        # temp = nx.Graph()
        g = nx.from_numpy_array(cmat_array, create_using=nx.Graph())
        nx.set_node_attributes(g, node_names, name="CLASS")
        return g

    def prep_SSE(self):
        if self.cmat is None:
            cmat = self.__get_contact_matrix()
            ss_ele = cmat.index.tolist()
            self.cmat = cmat
            self.ss_ele = ss_ele
        if self.orientation is None:
            orientation = self.__get_orientation_matrix()
            self.orientation = orientation
        if self.nxgraph is None:
            g = self.__prep_graph_obj(threshold=self.threshold)
            self.nxgraph = g

        cmat = self.cmat
        orientation = self.orientation
        self.params['raw'] = True
        return cmat, orientation
    
    def get_nxgraph(self):
        graph = self.nxgraph 
        return graph 
    

    def get_cmat(self):
        cmat, _ = self.prep_SSE()
        return cmat
        
    def get_ss_ele(self):
        _, _ = self.prep_SSE()
        ss_ele = self.ss_ele
        return ss_ele
    
    def get_orientation(self):
        _, orientation = self.prep_SSE()
        return orientation
    
    def __binarise(self, mat, threshold=None):
        if threshold==None:
            threshold = self.threshold
        thresholded = mat.copy(True)
        # thresholded = thresholded.clip_upper(threshold)
        thresholded[thresholded<=threshold]=1
        thresholded[thresholded>threshold]=0
        self.binarised_cmat = thresholded.to_numpy()
        return thresholded

    def cmat_binarise(self, threshold=None):
        '''Thresholds the df into a binary matrix according to threshold'''
        cmat, _ = self.prep_SSE()
        thresholded = self.__binarise(mat=cmat, threshold=threshold)
        self.adj = thresholded # saves as adj
        self.params['processed'] = True
        self.params['omulti'] = False
        self.params['chain'] = False
        return self

    def cmat_to_smat(self, sim_func):
        cmat, _ = self.prep_SSE()
        smat = sim_func(cmat)
        self.adj = smat  # saves as adj
        self.params['processed'] = True
        self.params['omulti'] = False
        self.params['chain'] = False
        return self
    
    def adj_to_omultiadj(self, types=False):
        if not self.params['omulti']:
            _, orientation = self.prep_SSE()
            adj_mat = np.asarray(self.adj)
            if not types: 
                types = [1., 2., 3.]
                # types = list(set(orientation.replace([np.inf], 0).astype(int).to_numpy().flatten()) - {0.})
            omultiadj = np.zeros((len(types), len(adj_mat), len(adj_mat)))
            for t in range(len(types)):
                # print(types[t])
                t_mat = (orientation==types[t])*adj_mat
                omultiadj[t, :len(adj_mat), :len(adj_mat)] = t_mat
            self.adj = omultiadj
            self.params['omulti'] = True
        else: 
            omultiadj = self.adj
        return self

    def add_chainlayer(self, chain_func):
        if chain_func==None:
            raise ValueError('Please input a chain_func to evaluate the chainlayer')
        if not self.params['chain']:
            cmat, _ = self.prep_SSE()
            # omultiadj = self.adj_to_omultiadj(types=False)
            self.adj_to_omultiadj(types=False)
            omultiadj = self.adj
            contact_o_multiadj = np.concatenate((omultiadj, np.array([chain_func(val=self.chainval)(cmat)])))
            self.adj = contact_o_multiadj
            self.params['chain'] = True
        else:
            contact_o_multiadj = self.adj
        return self
        
    def get_adj(self):
        return self.adj
    
    
    def assemble_strict(self, pid, reset=True):
        '''Strictest biologcial case'''
        self.set_pid(pid=pid)
        _, _ = self.prep_SSE()
        self.cmat_to_smat(sim_func=self.sim_func).adj_to_omultiadj(types=False).add_chainlayer(chain_func=self.chain_func)
        o_multi_adj = self.get_adj()
        sse_list = self.get_ss_ele()
        nxgraph = self.nxgraph
        binarised_cmat = self.binarised_cmat
        if reset:
            self.reset_SSE()
        return o_multi_adj, sse_list, nxgraph, binarised_cmat

    def assemble_binary(self, pid, reset=True):
        '''Minimally biologcial case'''
        self.set_pid(pid=pid)
        _, _ = self.prep_SSE()
        self.cmat_to_smat(sim_func=self.sim_func).adj_to_omultiadj(types=False).add_chainlayer(chain_func=self.chain_func)
        o_multi_adj = self.get_adj()
        sse_list = self.get_ss_ele()
        nxgraph = self.nxgraph
        binarised_cmat = self.binarised_cmat
        if reset:
            self.reset_SSE()
        return o_multi_adj, sse_list, nxgraph, binarised_cmat


    def assemble_minimal(self, pid, reset=True):
        '''Minimally biologcial case'''
        self.set_pid(pid=pid)
        _, _ = self.prep_SSE()
        self.cmat_to_smat(sim_func=self.sim_func)#.add_chainlayer(chain_func=self.chain_func)
        self.adj = np.asarray(self.adj)
        adj = self.get_adj()
        sse_list = self.get_ss_ele()
        nxgraph = self.nxgraph
        binarised_cmat = self.binarised_cmat # binarised cmat can be the same, if thresholds are equal...
        if reset:
            self.reset_SSE()
        return adj, sse_list, nxgraph, binarised_cmat


# protein graph matching distance class:
class pair_PGMD:
    def __init__(self, assembly_func):
        # self.pids = None # pair of ids
        # self.fpathparams = [fpathprefix, fpathext]
        # self.opathparams = [opathprefix, opathext] 
        self.threshold = 7.5
        self.assembly_func = assembly_func
        self.container = None
        param_fields = ['container', 'layouts']
        self.params = {i:False for i in param_fields}
        self.sorted_ids = None
        self.output = {}
        self.layouts = None

    def prep_pair_params(self, pids):
        if not self.params['container']:
            # self.ids = pids # pair of ids
            # pids = self.ids
            o_multi_adj_1, sse_list_1, nxgraph_1, bcmat_1 = self.assembly_func(pid=pids[0], reset=True)
            o_multi_adj_2, sse_list_2, nxgraph_2, bcmat_2 = self.assembly_func(pid=pids[1], reset=True)
            adjs = [o_multi_adj_1, o_multi_adj_2]
            sse_lists = [sse_list_1, sse_list_2]
            nx_graphs = [nxgraph_1, nxgraph_2]
            bcmats = [bcmat_1, bcmat_2]
            container = sorted(list(zip(pids, adjs, sse_lists, nx_graphs, bcmats)), key=lambda x: len(x[2]))
            self.container = container

            sorted_ids = [container[0][0], container[1][0]]
            self.sorted_ids = sorted_ids
            structures = [self.prep_structure(pid=ele[0] , sse_list=ele[2]) for ele in container]
            self.structures = structures
            node_cont = [self.get_node_types_lengths(sse_list=ele[2]) for ele in container]
            self.node_cont = node_cont
            self.params['container'] = True
        return self
    
    def reset_container(self):
        self.params['container'] = False
        self.container = None

    def reset_layouts(self):
        self.params['layouts'] = False
        self.layouts = None

    def reset_output(self):
        self.output = {}

    def refresh(self):
        self.reset_container()
        self.reset_layouts()
        self.sorted_ids = None
        self.output = {}
        # self.assembly_func.reset_SSE()
    
    def get_container(self, pids):
        self.prep_pair_params(pids=pids)
        return self.container
    
    def prep_structure(self, pid, sse_list):
        col_name = pid + '_structure'        
        structure = pd.DataFrame({col_name:(sse_list)}) 
        return structure
    
    def get_node_types_lengths(self, sse_list):
        '''Slicing the first element in the index names of the adjacency dataframe as the node type
        returns a dictionary between the numerical index and the node type.'''
        node_types = dict(zip(range(len(sse_list)), [i[0] for i in sse_list]))
        node_lengths = [int(i[1:]) for i in sse_list]
        return node_types, node_lengths

    def prep_sim_mat(self, alpha=0, gamma=0, use_lengths=False):
        # self.prep_pair_params(self, pids=pids)
        # sorted_ids = self.sorted_ids
        if alpha == 0: # meaningless value for the sim matrix... 
            self.sim_mat = None
            return self
        node_cont = self.node_cont 
        node_types_1, node_types_2 = [cont[0] for cont in node_cont]
        node_lengths_1, node_lengths_2 = [cont[1] for cont in node_cont]
        sim_mat = np.zeros((len(node_types_1), len(node_types_2)))
        sim_val=alpha

        for i in range(len(node_types_1)):
            for j in range(len(node_types_2)):
                if node_types_1[i] == node_types_2[j]:
                    lengths = [node_lengths_1[i], node_lengths_2[j]]
                    if use_lengths:
                        sim_val = length_similiarity(lengths=lengths, alpha=alpha, gamma=gamma) 
                    sim_mat[i,j] = sim_val
        self.sim_mat = sim_mat
        return self

    def run_gm(self, pids, alpha=0, gamma=0, use_lengths=False, match_args=None):
        self.prep_pair_params(pids=pids).prep_sim_mat(alpha=alpha, gamma=gamma, use_lengths=use_lengths)
        sim_mat = self.sim_mat
        if match_args==None:
            match_args = {'rng':0, 'padding':'naive', 'transport':True, 'n_init':1}
        match_args['S'] = sim_mat
        container = self.container 
        g_1 = container[0][1]
        g_2 = container[1][1]
        _, perm_inds, _, _ = graph_match(g_1, g_2, **match_args)
        g2_to_g1_node_mapping = self.get_node_mapping(perm_inds=perm_inds)

        self.output['perm_inds'] = perm_inds
        self.output['node_mapping'] = g2_to_g1_node_mapping
        return self.output
    
    def get_node_mapping(self, perm_inds):
        container = self.container
        # g_1 = container[0][1]
        g_2 = container[1][1]

        all_inds = list(range(g_2.shape[1]))
        not_matched = [i for i in all_inds if i not in perm_inds]
        full_inds = np.concatenate([perm_inds, not_matched])

        structures = self.structures 
        colnames = [df.columns[0].split('_structure')[0] + '_index' for df in structures] 
        structure_1_copy = structures[0].reset_index().rename(columns={'index':colnames[0]}).copy(True)
        structure_2_copy = structures[1].iloc[full_inds].reset_index().rename(columns={'index':colnames[1]}).copy(True)
        g2_to_g1_node_mapping = pd.concat([structure_1_copy, structure_2_copy], axis=1)
        g2_to_g1_node_mapping.replace({'':np.nan}, inplace=True)
        g2_to_g1_node_mapping.dropna(how='all', inplace=True)
        self.node_mapping = g2_to_g1_node_mapping
        return g2_to_g1_node_mapping
    

    def __compute_layouts(self, seed=0):
        if not self.params['layouts']:
            perm_inds = self.output['perm_inds']
            container = self.container
            graph_1 = container[0][3]
            graph_2 = container[1][3]
            # print(len(graph_1.nodes()))
            # print(len(graph_2.nodes()))
            node_mapping = dict(zip(graph_1.nodes(), np.array(graph_2.nodes())[perm_inds]))

            layout_1 = nx.spring_layout(graph_1, seed=seed)
            layout_2 = nx.spring_layout(graph_2, seed=seed)

            for key in layout_1: # iterate thorugh smaller graph 
                ismapped = node_mapping[key]
                if ismapped is not None:
                    layout_2[ismapped] = layout_1[key]
            layouts = [layout_1, layout_2]
            self.layouts = layouts
            self.params['layouts'] = True
        
    
    def get_layouts(self, seed=0):
        self.__compute_layouts(seed=seed)
        return self.layouts
    

    def get_layoutput(self, seed=0):
        container = self.container
        perm_inds = self.output['perm_inds']
        adj_1 = container[0][4]
        adj_2 = container[1][4]
        graph_1 = container[0][3]
        graph_2 = container[1][3]
        layouts = self.get_layouts(seed=seed)
        layoutput = distance_stats(adj_1=adj_1, adj_2=adj_2, perm_inds=perm_inds, directed=False)
    
        layoutput.update({'graph_1':graph_1, 'layout_1':layouts[0], 'graph_2':graph_2, \
                'layout_2':layouts[1], 'perm_inds':perm_inds,\
                        'adj_1':adj_1, 'adj_2':adj_2})
        
        return layoutput

        



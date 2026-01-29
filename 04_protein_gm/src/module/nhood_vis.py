import os
import time
import copy 
import numpy as np 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from graspologic.match import graph_match

# copied over from celegans/src/grmchd/nhood_vis

from .nhood import get_io_top_NM
from .gmd import matching_euclidean_distance, distance_stats


def df_to_graph(df, directed=True):
    ''' adds edges from dataframe df to a directed networkx graph '''
    if directed:
        g = nx.DiGraph()
    if not directed:
        g = nx.Graph()
    g.add_edges_from(df[['pre','post']].to_numpy())
    return g

def get_graphs(s_ids, df, node_number, edge_number, directed=True):
    '''Given s_ids, return the networkx graph of the top N, M edges and nodes from its neighbourhood via dataframe of edges df.'''
    graphs =[]
    for i in s_ids:
        graph = df_to_graph(get_io_top_NM(Id=i, df=df, N=edge_number, M=node_number), directed=directed)
        graphs.append(graph)
    return graphs


### 16/11/2023: added node types as marker shape
def get_graph_traces_plotly(G, seed=0, g=1, layout=None, labelling=False, directed=True, node_types=None):
    '''
    Outputs a scatter graph object of the neurons, and
    annotations of directed edges. 

    G : networkx graph
    layout : input layout of node to coordinate dictionary. If none, then spring layout used. 
    seed : used for graph plotting - spring layout random seed. 
    g : 1 for graph number 1; 2 for graph number 2 : used for labelling. 
    node_types : int or None. If none, they are drawn with the default shape: circle. 
            if int, then will take the label[:node_types+1] as the type and then draw different shape for each label. 
            max number of types ~ 50. 
    '''
    # initialise empty labels:
    extra_labelling =dict(zip(G.nodes(), ["" for _ in range(len(G))]))
    
    if g==1:
        ec = '#CC0000'
        nc = '#FFCCCC'
        g_label = 'G_1 '
    if g==2:
        ec = '#0000CC'
        nc = '#6666FF'
        g_label = 'G_2 '
    if type(g) == dict:
        ec = g['ec'] # '#800080'
        nc = g['nc'] #'#dda0dd'
        g_label = g['g_label']
    if labelling: 
        extra_labelling = nx.get_node_attributes(G, "CLASS")
    
    if not node_types:
        node_type_dict = dict()
    if type(node_types)==int:
        # the set of node types determined by the extra labelling: here we just take the node first node_types letters of the string. 
        types_set = set([i[:node_types+1] for i in extra_labelling.values()]) 
        types_dict = dict(zip(types_set, range(len(types_set))))
        node_type_dict = {key:types_dict[value[:node_types+1]] for key, value in extra_labelling.items()}
    
    edge_x = []
    edge_y = []

    if not layout:
        layout = nx.spring_layout(G, seed=seed)

    arrowhead_size = 0
    if directed:
        arrowhead_size = 2
    list_of_all_arrows = []
    for edge in G.edges():
        x0, y0 = layout[edge[1]] ## this is the source of the edge 
        x1, y1 = layout[edge[0]] ## this is the sink of the edge
        arrow = go.layout.Annotation(dict(
                        x=x0,
                        y=y0,
                        xref="x", yref="y",
                        text="",
                        showarrow=True,
                        axref="x", ayref='y',
                        ax=x1,
                        ay=y1,
                        arrowhead=arrowhead_size, 
                        arrowwidth=3,
                        arrowcolor=ec,)
                    )
        list_of_all_arrows.append(arrow)

    node_text = []
    node_x = []
    node_y = []
    marker_symbols = []

    for node in G.nodes():
        x, y = layout[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(g_label + str(node) + " " + extra_labelling[node])
        marker_symbols.append(node_type_dict.get(node, 0))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        # text='text',
        marker_symbol=marker_symbols,
        name=g_label,
        marker=dict(
            color=nc,
            size=20,
            opacity=1,
            line_width=1, 
            line_color=ec, 
            ))

    node_trace.text = node_text
    return node_trace, list_of_all_arrows


def plot_pl_from_graph(Graph, seed=0, iob=False):
    '''Plotly plot one graph given networkx graph Graph. '''

    node_trace, list_of_all_arrows = get_graph_traces_plotly(Graph, seed=seed, g=1, layout=None, iob=iob)
    fig = go.Figure(data=[node_trace],
                layout=go.Layout(
                    title='Main Graph',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.update_layout(annotations=list_of_all_arrows)
    return fig



def plot_single_graph(n_id, df, edge_number, node_number, seed=0):
    '''Plot one neighbourhood network with plotly given neuron id: n_id'''

    graph = get_graphs(s_ids=[n_id], df=df, node_number=node_number, edge_number=edge_number)[0]
    fig = plot_pl_from_graph(Graph=graph, seed=seed)
    fig.update_layout(
        title_text=f"{n_id} neighbourhood",
        xaxis_domain=[0.05, 1.0]
    )
    fig.show()

def compute_layouts(graphs, seed, matching=None, directed=True):
    '''
    
    Computes the layout of two graphs given initial seed using spring_layout and 
    mapping node from graphs[0] to graphs[1] using graph matching. 

    Requires that graphs[0] is the smaller graph than graph[1]

    if ged==True, then calculates the approx graph edit distance using the matching procedure. 
    
    '''

    srtd = sorted([i for i in graphs], key=lambda x:len(x))

    graph_1 = srtd[0] # smaller graph
    adj_1 = nx.adjacency_matrix(srtd[0]).todense()
    graph_2 = srtd[1] # larger graph
    adj_2 = nx.adjacency_matrix(srtd[1]).todense()

    layout_1 = nx.spring_layout(graph_1, seed=seed)
    layout_2 = nx.spring_layout(graph_2, seed=seed)

    if matching is None:
        _, perm_inds, _, _ = graph_match(adj_1, adj_2, rng=0, padding='naive', transport=True)
    else:
        perm_inds = matching

    # mtrx_ij = adj_2[np.ix_(perm_inds, perm_inds)]


    # dict: if smaller graph --> node mapping exists:
    node_mapping = dict(zip(graph_1.nodes(), np.array(graph_2.nodes())[perm_inds]))
    # fnorm = np.linalg.norm(adj_1 - mtrx_ij) # distance: the frobenius norm between the matched matrices

    for key in layout_1: # iterate thorugh smaller graph 
        ismapped = node_mapping[key]
        if ismapped is not None:
            layout_2[ismapped] = layout_1[key]

    output = distance_stats(adj_1=adj_1, adj_2=adj_2, perm_inds=perm_inds, directed=directed)
    
    # test = ~((mtrx_ij - adj_1 <0.).any() * (mtrx_ij - adj_1 <0.).any() )
    # # if the matched graph_2 or graph_1 is a subgraph of the other, then ged calculation is guaranteed to be exact. 
    # print('Exact subgraph?', test)
    # edge_d_b_ap = sum(abs(adj_1 - mtrx_ij)).sum() # edge edit distance between graph_1 and matched graph_2
    # edge_d_a_ap = abs(sum(adj_2).sum() - sum(mtrx_ij).sum()) # edge number differnce between graph_2 and matched graph_2
    # node_d_a_ap = abs(len(adj_2) - len(mtrx_ij)) # node number difference between graph_2 and matched graph_2
    # graph_edit_distance = edge_d_b_ap + edge_d_a_ap + node_d_a_ap # this is always at least an overestimate 
    # # print(graph_edit_distance)
    # # print('edge_d_b_ap',edge_d_b_ap )
    # # print('edge_d_a_ap',edge_d_a_ap )
    # # print('node_d_a_ap',node_d_a_ap )
    # euclidean_distance = matching_euclidean_distance(adj_1=adj_1, adj_2=adj_2, perm_inds=perm_inds)

    # # return graph_1, layout_1, graph_2, layout_2, ed, graph_edit_distance

    # output = {'graph_1':graph_1, 'layout_1':layout_1, 'graph_2':graph_2, \
    #           'layout_2':layout_2, 'ed':ed, 'graph_edit_distance':graph_edit_distance,\
    #             'euclidean':euclidean_distance, 'perm_inds':perm_inds,\
    #                 'adj_1':adj_1, 'adj_2':adj_2}

    output.update({'graph_1':graph_1, 'layout_1':layout_1, 'graph_2':graph_2, \
              'layout_2':layout_2, 'perm_inds':perm_inds,\
                    'adj_1':adj_1, 'adj_2':adj_2})
    return output
    # return graph_1, layout_1, graph_2, layout_2, ed


# needs rewriting:

# def get_graphs_and_layouts(s_ids, seed, df,N, M, labelling=False):

#     '''
#     s_ids : list of two node IDs.

    
#     '''

#     graphs = get_graphs(s_ids=s_ids, df=df, node_number=M, edge_number=N)

#     # edit_paths= []
#     # for v in nx.optimize_edit_paths(graphs[0], graphs[1], node_match=spec_nmatch): 
#     #     edit_paths.append(v)

#     graph_1, layout_1, graph_2, layout_2, fnorm = compute_layouts(graphs=graphs, seed=seed)
#     return graph_1, layout_1, graph_2, layout_2, fnorm

# def plot_graph_overlay(A_id, B_id, df, N, M, seed=0, labelling=False):
#     '''
    
#     Plotly plots A_id and B_id neuron's neighbourhoods. 
#     The layout starts with networkx's spring layout and the two neighbourhood neurons 
#     are matched according to the exact Graph Edit Distance calculation. 

#     iob : True if in/out/both labellings are used for the GED calculation and as hover data in the plot. 

#     '''

#     A_id_neighbourhood = set(list(np.hstack(get_io_top_NM(A_id, df, N=N, M=M)[['pre', 'post']].to_numpy())))
#     B_id_neighbourhood = set(list(np.hstack(get_io_top_NM(B_id, df, N=N, M=M)[['pre', 'post']].to_numpy())))

#     if len(A_id_neighbourhood) > len(B_id_neighbourhood):
#         s_ids = [A_id, B_id] # larger id first 
#     else:
#         s_ids = [B_id, A_id]

#     g_1, layout_1, g_2, layout_2, fnorm = get_graphs_and_layouts(s_ids, N=N, M=M, seed=seed, iob=labelling, df=df)


#     nodes_1, edges_1 = get_graph_traces_plotly(g_1,g=1, seed=seed, layout=layout_1, labelling=labelling)
#     nodes_2, edges_2 = get_graph_traces_plotly(g_2,g=2, seed=seed, layout=layout_2, labelling=labelling)

#     fig = go.Figure(data=[nodes_1, nodes_2],
#             layout=go.Layout(
#                 title='Main Graph',
#                 titlefont_size=16,
#                 showlegend=True,
#                 hovermode='closest',
#                 margin=dict(b=20,l=5,r=5,t=40),
#                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
#                 )
#     # fig.update_traces(marker=dict(
#     #     color='Blue', opacity=1))

#     fig.update_layout(
#     updatemenus=[
#         dict(
#             type="buttons",
#             direction="right",
#             active=0,
#             x=1,
#             y=-0.2,
#             buttons=list([
#                 dict(label=f"G_1 edges",
#                      method="update",
#                      args=[{"visible": [True, True, False, False]},
#                            {
#                             "annotations": edges_1}]),
#                 dict(label=f"G_2 edges",
#                      method="update",
#                      args=[{"visible": [True, True, False, False]},
#                            {
#                             "annotations": edges_2}]),
#                 dict(label="Both edges",
#                      method="update",
#                      args=[{"visible": [True, True, True, True]},
#                            {
#                             "annotations": edges_1 + edges_2}]),
#                 dict(label="None",
#                      method="update",
#                      args=[{"visible": [True, True, True, True]},
#                            {
#                             "annotations":[]}]),


#             ]),
#         )
#     ])
#     fig.update_layout(
#         title_text=f"G_1: {s_ids[0]} and G_2: {s_ids[1]}\nd={ed:.3f}",
#         xaxis_domain=[0.05, 1.0]
#     )
#     # edges_1.extend(edges_2)
#     # fig.update_layout(annotations=edges_1)
#     # fig.update_layout(annotations=edges_2)

#     fig.show()

def get_overlap_info_layout(output, directed=True):
    '''
    Gets the graph for which the edges overlap
    '''
    if directed:
        template = nx.DiGraph
    if not directed: 
        template = nx.Graph

    graph_1 = output['graph_1']
    graph_2 = output['graph_2']

    perm_inds = output['perm_inds']

    adj_1 = output['adj_1']
    adj_2 = output['adj_2']

    all_inds = list(range(len(adj_2)))


    adj_1_transformed = np.zeros((len(adj_2), len(adj_2))) # initialise larger container
    adj_1_transformed[:len(adj_1), :len(adj_1)] = adj_1 

    not_matched = [i for i in all_inds if i not in perm_inds]

    if len(not_matched)<1:
        full_inds=perm_inds
    else:
        full_inds = np.concatenate([perm_inds, not_matched])

    inv_inds = invert_indices(full_inds)

    adj_2_transformed = adj_2[np.ix_(full_inds, full_inds)]
 

    mtrx_ij = adj_2[np.ix_(perm_inds, perm_inds)]

    # edges that are unique to each graph:
    unique_edges_1 = (adj_1 - mtrx_ij > 0.5 ) * 1
    unique_edges_2 = (adj_2_transformed - adj_1_transformed > 0.5) * 1 ### this doesn't include the edges that are in the unmatched part of the graph
    
    unique_edges_2_ind_corrected = unique_edges_2[np.ix_(inv_inds, inv_inds)]
    
    common_edges = np.multiply(adj_1,mtrx_ij) 

    overlap_graph = nx.from_numpy_array(common_edges, create_using=template)
    unique_graph_1 = nx.from_numpy_array(unique_edges_1, create_using=template)
    unique_graph_2 = nx.from_numpy_array(unique_edges_2_ind_corrected, create_using=template)
    graphs = {'o_g':overlap_graph, 'u_g_1':unique_graph_1, 'u_g_2':unique_graph_2}

    return graphs




def invert_indices(inds):
    '''invert the array of indices such that i is on the inds[i]th element of output'''
    inv_inds_dict = dict(zip(inds, range(len(inds))))

    inv_inds = np.zeros((len(inds)))

    for i in range(len(inv_inds)):
        inv_inds[i] = inv_inds_dict[i]

    return inv_inds.astype(int)



def plot_two_graphs_overlay(cont_1, cont_2, seed=0, labelling=False,  directed=True):

    '''
    
    Plotly plots cont_1 and cont_2 graphs overlaying using graph matching. 

    cont_i['id'] is the id of the network
    cont_i['graph'] is the networkx graph object. 

    The layout starts with networkx's spring layout and the two graphs 
    are matched according to the graph matching algorithm. 

    '''


    srtd = sorted([cont_1, cont_2], key=lambda x:len(x['graph'])) # smallest first


    A_id = srtd[0]['id'] # smaller graph id
    B_id = srtd[1]['id'] # larger graph id

    A_graph= srtd[0]['graph']
    B_graph= srtd[1]['graph']

    s_ids = [A_id, B_id] # smaller id first. 

    graphs = [A_graph, B_graph]
    t0 = time.time()
    output = compute_layouts(graphs=graphs, seed=seed, directed=directed)

    graph_1 = output['graph_1']
    layout_1 = output['layout_1']
    graph_2 = output['graph_2']
    layout_2 = output['layout_2']
    fnorm = output['fnorm']
    ged = output['graph_edit_distance']
    euclidean_distance = output['euclidean_distance']
    isexact = output['isexact']

    print('time for calculation:', time.time()- t0)
    nodes_1, edges_1 = get_graph_traces_plotly(graph_1,g=1, seed=seed, layout=layout_1, labelling=labelling, directed=directed)
    nodes_2, edges_2 = get_graph_traces_plotly(graph_2,g=2, seed=seed, layout=layout_2, labelling=labelling, directed=directed)

    overlap_graphs = get_overlap_info_layout(output, directed=directed)

    # overlapping edges or unique edges:
    _, common_edges = get_graph_traces_plotly(overlap_graphs['o_g'],g={'ec':'#800080', 'nc':'#dda0dd', 'g_label':'common_edges'}, seed=seed, layout=layout_1, labelling=None, directed=directed)
    _, g_1_unique_edges = get_graph_traces_plotly(overlap_graphs['u_g_1'],g=1, seed=seed, layout=layout_1, labelling=None, directed=directed)
    _, g_2_unique_edges = get_graph_traces_plotly(overlap_graphs['u_g_2'],g=2, seed=seed, layout=layout_2, labelling=None, directed=directed)

    fig = go.Figure(data=[nodes_1, nodes_2],
            layout=go.Layout(
                title='Main Graph',
                titlefont_size=16,
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    # fig.update_traces(marker=dict(
    #     color='Blue', opacity=1))

    fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=1,
            y=-0.2,
            buttons=list([
                dict(label=f"G_1 edges",
                     method="update",
                     args=[{"visible": [True, True]},
                           {
                            "annotations": edges_1}]),
                dict(label=f"G_2 edges",
                     method="update",
                     args=[{"visible": [True, True]},
                           {
                            "annotations": edges_2}]),
                dict(label="Overlap",
                     method="update",
                     args=[{"visible": [True, True]},
                           {
                            "annotations": common_edges}]),
                dict(label="G_1 unique",
                     method="update",
                     args=[{"visible": [True, True]},
                           {
                            "annotations":g_1_unique_edges}]),
                dict(label="G_2 unique",
                     method="update",
                     args=[{"visible": [True, True]},
                           {
                            "annotations":g_2_unique_edges}]),
                dict(label="ALL",
                     method="update",
                     args=[{"visible": [True, True]},
                           {
                            "annotations":edges_1+edges_2}]),
            ]),
        )
    ])
    fig.update_layout(
        title_text=f"G_1: {s_ids[0]} and G_2: {s_ids[1]}, d={ged:.3f}",
        xaxis_domain=[0.05, 1.0]
    )

    full_fig = fig.full_figure_for_development(warn=False)
    xrange = full_fig.layout.xaxis.range
    yrange = full_fig.layout.yaxis.range
    fig.update_layout(xaxis_range=xrange, yaxis_range=yrange,
                        autosize=False,
                        width=800,
                        height=500,)


    fig.show()

    print('ids:', s_ids)
    print('sizes: (nodes, edges)', len(A_graph), len(A_graph.edges()), ', ', len(B_graph), len(B_graph.edges()))
    print('Exact subgraph?', isexact)
    print(f'square F-norm: {fnorm**2:.3f}')  
    print('Euclidean distance:', euclidean_distance)
    print('edge overlap number:', len(overlap_graphs['o_g'].edges))
    print('G_1 unique edge number:', len(overlap_graphs['u_g_1'].edges))
    print('G_2 unique edge number:', len(overlap_graphs['u_g_2'].edges))
    


#### new 16/11/2023
def compute_layouts_from_cont(container, seed, matching=None, directed=True):
    '''
    container : 2 lists containing the Networkx graph object and (multi) adjacency matrix. 

    Computes the layout of two graphs given initial seed using spring_layout and 
    mapping node from graphs[0] to graphs[1] using graph matching. 

    Requires that graphs[0] is the smaller graph than graph[1]

    if ged==True, then calculates the approx graph edit distance using the matching procedure. 
    
    '''

    srtd = sorted([i for i in container], key=lambda x:len(x[0]))

    graph_1 = srtd[0][0] # smaller graph
    adj_1 = srtd[0][1]
    graph_2 = srtd[1][0] # larger graph
    adj_2 = srtd[1][1]

    layout_1 = nx.spring_layout(graph_1, seed=seed)
    layout_2 = nx.spring_layout(graph_2, seed=seed)

    if matching is None:
        _, perm_inds, _, _ = graph_match(adj_1, adj_2, rng=0, padding='naive', transport=True)
    else:
        perm_inds = matching

    # dict: if smaller graph --> node mapping exists:
    node_mapping = dict(zip(graph_1.nodes(), np.array(graph_2.nodes())[perm_inds]))

    for key in layout_1: # iterate thorugh smaller graph 
        ismapped = node_mapping[key]
        if ismapped is not None:
            layout_2[ismapped] = layout_1[key]

    output = distance_stats(adj_1=adj_1, adj_2=adj_2, perm_inds=perm_inds, directed=directed)
    
    output.update({'graph_1':graph_1, 'layout_1':layout_1, 'graph_2':graph_2, \
              'layout_2':layout_2, 'perm_inds':perm_inds,\
                    'adj_1':adj_1, 'adj_2':adj_2})
    return output

    

    
def plot_two_graphs_from_output(output, s_ids, directed=True, seed=0, labelling=True):
        
    graph_1 = output['graph_1']
    layout_1 = output['layout_1']
    graph_2 = output['graph_2']
    layout_2 = output['layout_2']
    fnorm = output['fnorm']
    ged = output['graph_edit_distance']
    euclidean_distance = output['euclidean_distance']
    isexact = output['isexact']

    # print('time for calculation:', time.time()- t0)
    nodes_1, edges_1 = get_graph_traces_plotly(graph_1,g=1, seed=seed, layout=layout_1, labelling=labelling, directed=directed, node_types=0)
    nodes_2, edges_2 = get_graph_traces_plotly(graph_2,g=2, seed=seed, layout=layout_2, labelling=labelling, directed=directed, node_types=0)

    overlap_graphs = get_overlap_info_layout(output, directed=directed)

    # overlapping edges or unique edges:
    _, common_edges = get_graph_traces_plotly(overlap_graphs['o_g'],g={'ec':'#800080', 'nc':'#dda0dd', 'g_label':'common_edges'}, seed=seed, layout=layout_1, labelling=None, directed=directed)
    _, g_1_unique_edges = get_graph_traces_plotly(overlap_graphs['u_g_1'],g=1, seed=seed, layout=layout_1, labelling=None, directed=directed)
    _, g_2_unique_edges = get_graph_traces_plotly(overlap_graphs['u_g_2'],g=2, seed=seed, layout=layout_2, labelling=None, directed=directed)

    fig = go.Figure(data=[nodes_1, nodes_2],
            layout=go.Layout(
                title='Main Graph',
                titlefont_size=16,
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    # fig.update_traces(marker=dict(
    #     color='Blue', opacity=1))

    fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=1,
            y=-0.2,
            buttons=list([
                dict(label=f"G_1 edges",
                        method="update",
                        args=[{"visible": [True, True]},
                            {
                            "annotations": edges_1}]),
                dict(label=f"G_2 edges",
                        method="update",
                        args=[{"visible": [True, True]},
                            {
                            "annotations": edges_2}]),
                dict(label="Overlap",
                        method="update",
                        args=[{"visible": [True, True]},
                            {
                            "annotations": common_edges}]),
                dict(label="G_1 unique",
                        method="update",
                        args=[{"visible": [True, True]},
                            {
                            "annotations":g_1_unique_edges}]),
                dict(label="G_2 unique",
                        method="update",
                        args=[{"visible": [True, True]},
                            {
                            "annotations":g_2_unique_edges}]),
                dict(label="ALL",
                        method="update",
                        args=[{"visible": [True, True]},
                            {
                            "annotations":edges_1+edges_2}]),
            ]),
        )
    ])
    fig.update_layout(
        title_text=f"G_1: {s_ids[0]} and G_2: {s_ids[1]}, d={ged:.3f}",
        xaxis_domain=[0.05, 1.0]
    )

    full_fig = fig.full_figure_for_development(warn=False)
    xrange = full_fig.layout.xaxis.range
    yrange = full_fig.layout.yaxis.range
    fig.update_layout(xaxis_range=xrange, yaxis_range=yrange,
                        autosize=False,
                        width=800,
                        height=500,)


    fig.show()

    print('ids:', s_ids)
    print('sizes: (nodes, edges)', len(graph_1), len(graph_1.edges()), ', ', len(graph_2), len(graph_2.edges()))
    print('Exact subgraph?', isexact)
    print(f'square F-norm: {fnorm**2:.3f}')  
    print('Euclidean distance:', euclidean_distance)
    print('edge overlap number:', len(overlap_graphs['o_g'].edges))
    print('G_1 unique edge number:', len(overlap_graphs['u_g_1'].edges))
    print('G_2 unique edge number:', len(overlap_graphs['u_g_2'].edges))

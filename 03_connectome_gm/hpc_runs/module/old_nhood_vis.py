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


from .old_nhood import get_io_top_NM

def get_graph_traces_plotly(G, seed=0, g=1, layout=None, iob=False):
    '''
    Outputs a scatter graph object of the neurons, and
    annotations of directed edges. 

    G : networkx graph
    layout : input layout of node to coordinate dictionary. If none, then spring layout used. 
    seed : used for graph plotting - spring layout random seed. 
    g : 1 for graph number 1; 2 for graph number 2 : used for labelling. 


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
    if iob: 
        extra_labelling = nx.get_node_attributes(G, "CLASS")
        
    edge_x = []
    edge_y = []

    if not layout:
        layout = nx.spring_layout(G, seed=seed)

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
                        arrowhead=2, 
                        arrowwidth=3,
                        arrowcolor=ec,)
                    )
        list_of_all_arrows.append(arrow)

    node_text = []
    node_x = []
    node_y = []

    for node in G.nodes():
        x, y = layout[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(g_label + str(node) + " " + extra_labelling[node])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        # text='text',
        name=g_label,
        marker=dict(
            color=nc,
            size=20,
            opacity=1,
            line_width=1, 
            line_color=ec))

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

def df_to_graph(df):
    ''' adds edges from dataframe df to a directed networkx graph '''
    dg = nx.DiGraph()
    dg.add_edges_from(df[['pre','post']].to_numpy())
    return dg


def get_graphs(s_ids, df, node_number, edge_number, iob=False):
    '''Given s_ids, return the networkx graph of the top N, M edges and nodes from its neighbourhood via dataframe of edges df.'''
    graphs =[]
    for i in s_ids:
        graph = df_to_graph(get_io_top_NM(Id=i, df=df, N=edge_number, M=node_number))
        graphs.append(graph)
    return graphs


def plot_single_graph(n_id, df, edge_number, node_number, seed=0):
    '''Plot one neighbourhood network with plotly given neuron id: n_id'''

    graph = get_graphs(s_ids=[n_id], df=df, node_number=node_number, edge_number=edge_number)[0]
    fig = plot_pl_from_graph(Graph=graph, seed=seed)
    fig.update_layout(
        title_text=f"{n_id} neighbourhood",
        xaxis_domain=[0.05, 1.0]
    )
    fig.show()


def compute_layouts(graphs, seed, ged=False):
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

    _, perm_inds, _, _ = graph_match(adj_1, adj_2, rng=0, padding='naive', transport=True)
    mtrx_ij = adj_2[np.ix_(perm_inds, perm_inds)]

    # dict: if smaller graph --> node mapping exists:
    node_mapping = dict(zip(graph_1.nodes(), np.array(graph_2.nodes())[perm_inds]))
    ed = np.linalg.norm(adj_1 - mtrx_ij) # distance: the frobenius norm between the matched matrices

    # node_mapping=dict(zip())
    # ed = edit_paths[-1][-1]

    for key in layout_1: # iterate thorugh smaller graph 
        ismapped = node_mapping[key]
        if ismapped is not None:
            layout_2[ismapped] = layout_1[key]
    if ged:
        # print(mtrx_ij - adj_1 )
        # print(adj_1 - mtrx_ij)
        test = ~((mtrx_ij - adj_1 <0.).any() * (mtrx_ij - adj_1 <0.).any() )
        # if the matched graph_2 or graph_1 is a subgraph of the other, then ged calculation is guaranteed to be exact. 
        print('Exact subgraph?', test)
        edge_d_b_ap = sum(abs(adj_1 - mtrx_ij)).sum() # edge edit distance between graph_1 and matched graph_2
        edge_d_a_ap = abs(sum(adj_2).sum() - sum(mtrx_ij).sum()) # edge number differnce between graph_2 and matched graph_2
        node_d_a_ap = abs(len(adj_2) - len(mtrx_ij)) # node number difference between graph_2 and matched graph_2
        graph_edit_distance = edge_d_b_ap + edge_d_a_ap + node_d_a_ap # this is always at least an overestimate 
        # print(graph_edit_distance)
        # print('edge_d_b_ap',edge_d_b_ap )
        # print('edge_d_a_ap',edge_d_a_ap )
        # print('node_d_a_ap',node_d_a_ap )

        return graph_1, layout_1, graph_2, layout_2, ed, graph_edit_distance

    return graph_1, layout_1, graph_2, layout_2, ed


def get_graphs_and_layouts(s_ids, seed, df,N, M, iob=False):

    '''
    s_ids : list of two node IDs.

    
    '''

    graphs = get_graphs(s_ids=s_ids, iob=iob, df=df, node_number=M, edge_number=N)

    # edit_paths= []
    # for v in nx.optimize_edit_paths(graphs[0], graphs[1], node_match=spec_nmatch): 
    #     edit_paths.append(v)

    graph_1, layout_1, graph_2, layout_2, ed = compute_layouts(graphs=graphs, seed=seed)
    return graph_1, layout_1, graph_2, layout_2, ed

def plot_graph_overlay(A_id, B_id, df, N, M, seed=0, iob=False):
    '''
    
    Plotly plots A_id and B_id neuron's neighbourhoods. 
    The layout starts with networkx's spring layout and the two neighbourhood neurons 
    are matched according to the exact Graph Edit Distance calculation. 

    iob : True if in/out/both labellings are used for the GED calculation and as hover data in the plot. 

    '''

    A_id_neighbourhood = set(list(np.hstack(get_io_top_NM(A_id, df, N=N, M=M)[['pre', 'post']].to_numpy())))
    B_id_neighbourhood = set(list(np.hstack(get_io_top_NM(B_id, df, N=N, M=M)[['pre', 'post']].to_numpy())))

    if len(A_id_neighbourhood) > len(B_id_neighbourhood):
        s_ids = [A_id, B_id] # larger id first 
    else:
        s_ids = [B_id, A_id]

    g_1, layout_1, g_2, layout_2, ed = get_graphs_and_layouts(s_ids, N=N, M=M, seed=seed, iob=iob, df=df)


    nodes_1, edges_1 = get_graph_traces_plotly(g_1,g=1, seed=seed, layout=layout_1, iob=iob)
    nodes_2, edges_2 = get_graph_traces_plotly(g_2,g=2, seed=seed, layout=layout_2, iob=iob)

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
                     args=[{"visible": [True, True, False, False]},
                           {
                            "annotations": edges_1}]),
                dict(label=f"G_2 edges",
                     method="update",
                     args=[{"visible": [True, True, False, False]},
                           {
                            "annotations": edges_2}]),
                dict(label="Both edges",
                     method="update",
                     args=[{"visible": [True, True, True, True]},
                           {
                            "annotations": edges_1 + edges_2}]),
                dict(label="None",
                     method="update",
                     args=[{"visible": [True, True, True, True]},
                           {
                            "annotations":[]}]),


            ]),
        )
    ])
    fig.update_layout(
        title_text=f"G_1: {s_ids[0]} and G_2: {s_ids[1]}\nd={ed:.3f}",
        xaxis_domain=[0.05, 1.0]
    )
    # edges_1.extend(edges_2)
    # fig.update_layout(annotations=edges_1)
    # fig.update_layout(annotations=edges_2)

    fig.show()


def plot_two_graphs_overlay(nhood_1, nhood_2, seed=0, iob=False):

    '''
    
    Plotly plots nhood_1 and nhood_2 neighbourhoods. 

    nhood_i['id'] is the neuron id
    nhood_i['neighbourhood'] is the networkx graph object. 

    The layout starts with networkx's spring layout and the two neighbourhood neurons 
    are matched according to the graph matching algorithm. 

    '''


    srtd = sorted([nhood_1, nhood_2], key=lambda x:len(x['neighbourhood'])) # smallest first


    A_id = srtd[0]['id'] # smaller neighbourhood id
    B_id = srtd[1]['id'] # larger neighbourhood id

    A_nhood= srtd[0]['neighbourhood']
    B_nhood= srtd[1]['neighbourhood']

    s_ids = [A_id, B_id] # smaller id first. 

    graphs = [A_nhood, B_nhood]
    t0 = time.time()
    graph_1, layout_1, graph_2, layout_2, ed, ged = compute_layouts(graphs=graphs, seed=seed, ged=True)
    print('time for calculation:', time.time()- t0)
    nodes_1, edges_1 = get_graph_traces_plotly(graph_1,g=1, seed=seed, layout=layout_1, iob=iob)
    nodes_2, edges_2 = get_graph_traces_plotly(graph_2,g=2, seed=seed, layout=layout_2, iob=iob)

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
                     args=[{"visible": [True, True, False, False]},
                           {
                            "annotations": edges_1}]),
                dict(label=f"G_2 edges",
                     method="update",
                     args=[{"visible": [True, True, False, False]},
                           {
                            "annotations": edges_2}]),
                dict(label="Both edges",
                     method="update",
                     args=[{"visible": [True, True, True, True]},
                           {
                            "annotations": edges_1 + edges_2}]),
                dict(label="None",
                     method="update",
                     args=[{"visible": [True, True, True, True]},
                           {
                            "annotations":[]}]),


            ]),
        )
    ])
    fig.update_layout(
        title_text=f"G_1: {s_ids[0]} and G_2: {s_ids[1]}, d={ged:.3f}",
        xaxis_domain=[0.05, 1.0]
    )

    # full_fig = fig.full_figure_for_development(warn=False)
    # xrange = full_fig.layout.xaxis.range
    # yrange = full_fig.layout.yaxis.range
    # fig.update_layout(xaxis_range=xrange, yaxis_range=yrange,
    #                     autosize=False,
    #                     width=800,
    #                     height=500,)


    fig.show()

    print('ids:', s_ids)
    print('sizes: (nodes, edges)', len(A_nhood), len(A_nhood.edges()), ', ', len(B_nhood), len(B_nhood.edges()))
    print(f'square F-norm: {ed**2:.3f}')  
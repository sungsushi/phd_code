import numpy as np 
import pandas as pd
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

from scipy.cluster.hierarchy import fcluster, to_tree, dendrogram, linkage
from matplotlib.pyplot import cm


def dendrogram_clustering(Z, labels, clusters, label_dict=None, title_str='', t=0, save=False):

    '''Plot dendrogram given linkage Z, labels and clusters '''

    size = len(labels)*60/400 
    plt.figure(figsize=(6,size))
    only_clusters = set(np.hstack(clusters))
    if label_dict:
        # clstr_labels = [i + " : " + str(label_dict.get(i)) if i in only_clusters else i for i in labels]
        clstr_labels = [i + " : " + str(label_dict.get(i)) for i in labels]

    else:
        clstr_labels=labels
    dendrograms = dendrogram(Z, labels=clstr_labels, get_leaves=True, orientation='left', color_threshold=t);


    colors = cm.rainbow(np.linspace(0,1,len(clusters)))
    ax = plt.gca()



    ylbls = ax.get_ymajorticklabels()
    for i in range(len(colors)):
        clstr = clusters[i]
        # print(clstr)
        for lbl in ylbls:
            label_colors = {'True' : colors[i], 'False' : 'black'}
            id_name = lbl.get_text().split(" : ")[0]
            is_clstr = id_name in clstr

            # print(lbl.get_text())
            # lbl.set_text(hemitype[5:])

            if is_clstr:
                lbl.set_color(label_colors[str(is_clstr)])


    plt.yticks(fontsize=10)
    plt.title(title_str)
    plt.tight_layout()
    if save:
        plt.savefig(f'{save}', bbox_inches='tight')
    plt.show()
    
def get_Z(vectors, metric='euclidean', method='ward'):

    combined = pd.concat(vectors).fillna(1e-10)

    ind_to_id = combined.index.values

    Z = linkage(combined, metric=metric, method=method) ####### linkage
    return Z, ind_to_id

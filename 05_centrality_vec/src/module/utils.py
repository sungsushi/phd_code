
import pandas as pd
from matplotlib.pyplot import cm
from scipy.cluster.hierarchy import to_tree, dendrogram
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.colors as mcolors

def dendrogram_clustering(Z, labels, clusters, label_dict=None, title_str='', t=0, save=False, axrange=None):

    '''Plot dendrogram given linkage Z, labels and clusters '''

    size = max([len(labels)*60/400, 5])
    plt.figure(figsize=(6,size))
    # only_clusters = set(np.hstack(clusters))
    if label_dict:
        # clstr_labels = [i + " : " + str(label_dict[i]) if i in only_clusters  else i for i in labels]
        clstr_labels = [i + " : " + str(label_dict.get(i, '')) for i in labels]

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
    plt.xlim(axrange, 0)

    plt.tight_layout()
    if save:
        plt.savefig(f'{save}', bbox_inches='tight')
    plt.show()


def arr_from_arrstring(string):
    elements = string[string.find('[') + 1 : string.find(']')]
    # print(elements)
    # Convert to a list of integers
    array = np.array([float(x) for x in elements.split(',') if x.strip()])

    return array


def lighter_shades(hex_color, n_shades=4):
    rgb = mcolors.hex2color(hex_color)
    shades = [mcolors.to_hex(rgb)]
    for i in range(1, n_shades+1):
        factor = 1 - (i * 0.15)
        lighter = tuple(1 - (1 - c) * factor for c in rgb)
        shades.append(mcolors.to_hex(lighter))
    return shades

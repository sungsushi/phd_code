import numpy as np
import pandas as pd

def prep_grn_data():
    data = pd.read_csv('../../data/grn/AtRegNet.csv', index_col=0)
    df =  pd.DataFrame(data.reset_index().iloc[:, :-4].to_numpy(), columns= data.columns[:-3])
    df['TFLocus'] = df['TFLocus'].str.strip().str.upper()
    df['TargetLocus'] = df['TargetLocus'].str.strip().str.upper()

    g_reg_edges = df[['TFLocus', 'TargetLocus']] # rename TFs as "pre" edge, regulated genes as "post" edge
    g_reg_edges.columns = ['pre', 'post'] 
    g_reg_edges = g_reg_edges.dropna().drop_duplicates() # no duplicate edges

    f = open('../../data/grn/gene_association.tair')
    data = f.readlines()
    data = data[5:]
    split = [x.rstrip("\n") for x in data]
    split = [x.split("\t") for x in split]
    gene_order = [i[-7].split('|')[0] for i in split]
    GO_term_order = [ i[4] for i in split]
    GO_bipartite_df = pd.DataFrame({'gene':gene_order, 'GO':GO_term_order}).drop_duplicates().reset_index(drop=True) # get rid of duplicated entries
    return g_reg_edges, GO_bipartite_df


def prep_go_meta():
    f = open('../../data/grn/ATH_GO_GOSLIM.txt')
    data = f.readlines()
    data = data[4:]

    split = [x.rstrip("\n") for x in data]
    split = [x.split("\t") for x in split]

    gene_order = [i[0] for i in split]
    GO_term_order = [i[5] for i in split]
    GO_term_meta_order = [i[4] for i in split]
    GO_slim_term_order = [i[8] for i in split]

    GO_meta_df = pd.DataFrame({'gene':gene_order,'GO_term':GO_term_order, 'GO_meta':GO_term_meta_order, 'GO_slim_term':GO_slim_term_order})
    go_meta_df = GO_meta_df[['GO_term', 'GO_meta']].drop_duplicates().reset_index(drop=True)
    return go_meta_df

def prep_tf_fam_dict():
    file = open('../../data/grn/tf_family.csv')
    data = file.readlines()
    colnames = data[0].split(',')[:2]
    family_data = [i.split(',')[:2] for i in data[1:]]

    tfs_label_df = pd.DataFrame(family_data, columns=colnames)
    tfs_label_df['TF Locus ID'] = tfs_label_df['TF Locus ID'].str.upper()
    tf_family_dict= tfs_label_df.set_index('TF Locus ID')['TF Family Name'].to_dict()
    return tf_family_dict



#### 


def prep_recipe_data():
        
    # edge data:
    f = open('../../data/recipe/recipe_ingredients_bipartite')
    read_edges = f.readlines()
    read_edges = [x.rstrip("\n") for x in read_edges]
    read_edges = [x.split("\t") for x in read_edges]
    read_edges = np.array(read_edges)
    rn_df = pd.DataFrame()
    rn_df['ingredient'] = read_edges[:,1]
    rn_df['r_id'] = read_edges[:,0]

    # meta data:
    f = open('../../data/recipe/cuisine_recipe')
    read_meta = f.readlines()
    read_meta = [x.rstrip("\n") for x in read_meta]
    read_meta = [i for i in read_meta if i != '@']
    read_meta = [x.split("\t") for x in read_meta]
    meta_list = []
    for i in read_meta:
        cuisine = i[0]
        r_ids = i[1:]
        update = [(r_id, cuisine) for r_id in r_ids]
        meta_list.extend(update)
    meta_df = pd.DataFrame(meta_list, columns=['r_id', 'cuisine'])

    # ingredient filtering: 
    rn_df_filtered = rn_df.copy(True)
    ingredient_list = list(set(rn_df.ingredient.dropna().unique()) - {''})
    rn_df_filtered = rn_df_filtered[rn_df_filtered.ingredient.isin(ingredient_list)]

    # filtering data not present as a node in the network:
    meta_set = set(meta_df.r_id) #set([i for i in meta_dict.keys()])
    df_set = set(rn_df_filtered.r_id)

    keep_ids =  df_set & meta_set


    meta_df_filtered = meta_df.copy(True)
    meta_df_filtered = meta_df_filtered[meta_df_filtered.r_id.isin(list(keep_ids))]

    rn_df_filtered = rn_df_filtered[rn_df_filtered.r_id.isin(list(keep_ids))]

    return rn_df_filtered, meta_df_filtered



# def prep_recipe_meta():
#     f = open('../data/recipe/cuisine_recipe')
#     read_meta = f.readlines()

#     read_meta = [x.rstrip("\n") for x in read_meta]
#     read_meta = [i for i in read_meta if i != '@']
#     read_meta = [x.split("\t") for x in read_meta]

#     meta_list = []
#     for i in read_meta:
#         cuisine = i[0]
#         r_ids = i[1:]
#         update = [(r_id, cuisine) for r_id in r_ids]
#         meta_list.extend(update)

#     r_c_df = pd.DataFrame(meta_list, columns=['r_id', 'cuisine'])
#     return r_c_df


def prep_foodweb_data():
        
    # edge data:
    f = open('../../data/foodweb/florida.bare_upto122_fullnames')
    edges = f.readlines()
    string_split = np.array([i.split(' ') for i in edges])
    fw_df = pd.DataFrame()
    fw_df['prey'] = string_split[:,0]
    fw_df['predator'] = string_split[:,1]
    fw_df['predator'] = fw_df['predator'].apply(lambda x: x[:-1])


    # meta data:
    f = open('../../data/foodweb/florida.bare_upto122_fullnames.newcat.terms')
    meta = f.readlines()
    name_split = np.array([i.split(',')[0] for i in meta])
    broad_cat = np.array([i.split(',')[-1][:-1] for i in meta])
    meta_dict = dict(zip(name_split, broad_cat))
    meta_df = pd.DataFrame()
    meta_df['node'] = name_split
    meta_df['type'] = broad_cat

    type_splt = [i.split(',')[1] for i in meta]
    type_splt = [x.rstrip("\n") for x in type_splt]
    meta_fine_df = pd.DataFrame()
    meta_fine_df['node'] = name_split
    meta_fine_df['type'] = type_splt


    # filtering data not present as a node in the network:
    meta_set = set([i for i in meta_dict.keys()])
    df_set = set(list(np.hstack(fw_df.to_numpy())))

    not_pred_or_prey = meta_set - df_set
    meta_df_filtered = meta_df.copy(True)
    meta_df_filtered = meta_df_filtered[~meta_df_filtered.node.isin(list(not_pred_or_prey))]

    meta_fine_df_filtered = meta_fine_df[~meta_fine_df.node.isin(list(not_pred_or_prey))]

    a = [i.split(',') for i in meta]
    b = [[x.rstrip("\n") for x in i] for i in a]
    c = [[x[0], x[-1]] if x[-1] not in ['Fish', 'Birds'] else [x[0], x[1]] for x in b ] # we want finer labels for small, medium, large fish...

    meta_fine_df_2 = pd.DataFrame(c, columns=['node', 'type'])
    inds = meta_fine_df_2.loc[meta_fine_df_2.node.isin(['Other_Pelagic_Fishes', 'Other_Demersal_Fishes'])].index.values # these are all other fish - separate category...
    meta_fine_df_2.loc[inds, 'type'] = 'Fish'

    meta_set = set(meta_fine_df_2.node.tolist())
    df_set = set(list(np.hstack(fw_df.to_numpy())))

    not_pred_or_prey = meta_set - df_set

    meta_fine_df_2_filtered = meta_fine_df_2.copy(True)
    meta_fine_df_2_filtered = meta_fine_df_2_filtered[~meta_fine_df_2_filtered.node.isin(list(not_pred_or_prey))]


    return fw_df, meta_df_filtered, meta_fine_df_filtered, meta_fine_df_2_filtered

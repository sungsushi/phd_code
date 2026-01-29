import numpy as np 
import matplotlib.pyplot as plt
from fvec.fvec import bipartite_cooarray, adjacency_cooarray, csr_row_norm, csr_col_norm
import pandas as pd
import scipy as sp

def get_entropy(vector, delta=1e-8):
    '''
    Gets the shannon entropy of a vector whose elements are probabilities. Ignores zeros.
    i.e. vector.sum() = 1

    Even with less than ideal machine accuracy, entropy contribution of v --> 0^+ approaches zero.  
    '''
    if vector.isnull().values.all(): # if the vector has no contributions, then return np.nan
        return np.nan 

    v = vector[vector > delta].values
    entropy = -sum(v * np.log(v))
    
    return entropy


def insert_line_break(s):
    if len(s) > 50:
        # Find the last space before or at the 55th character
        break_pos = s.rfind(' ', 0, 50)
        if break_pos == -1:
            # No space found before 55 — fall back to hard break
            break_pos = 50
        return s[:break_pos] + '\n' + s[break_pos+1:]
    return s




def make_GO_table(data, ax, highlight_row=None):
    """
    Create a matplotlib table displaying GO terms and descriptions.
    
    Args:
        data: Table data rows
        ax: Matplotlib axes object
        highlight_row: Optional row index to highlight
    """
    # fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')

    # Create the table
    table = ax.table(
        cellText=data,
        colLabels=['GO terms', 'Description'],
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)

    col_widths = [0.2,0.8]  # fractions of figure width
    # nrows, ncols = df.shape

    for i in range(len(data)+1):
        for j in range(2):
            cell = table[i, j]

            cell.set_width(col_widths[j])
            cell.set_edgecolor('black')


    GO_col = 0  # "Dist"
    desc_col = 1

    for row in range(len(data)+1):
        GO_col_cell = table[(row, GO_col)]  # +1 to skip header row
        text = GO_col_cell.get_text()
        text.set_fontweight('bold')
        GO_col_cell = table[(row, desc_col)]  # +1 to skip header row
        GO_col_cell.get_text().set_ha('left')
        desc_col_cell = table[(row, desc_col)]
        text = desc_col_cell.get_text().get_text()
        if '\n' in text:
            highlight_row = row

    if highlight_row is not None:
        # print(highlight_row)
        for col in range(2):  # Number of columns
            table[(highlight_row, col)].set_height(0.1) 

    # Optional: bold the column header too
    n_cols = 2
    for col in range(n_cols):
        header_cell = table[(0, col)]
        header_cell.get_text().set_fontweight('bold')    # plt.show()
        header_cell.get_text().set_fontsize(15)
    # return fig



def find_branch_point(Z, leaf_indices):
    """
    Find the branch point (cluster) in a hierarchical clustering linkage matrix Z that contains all specified leaf indices.
    Returns the cluster index and its height (distance), or (None, None) if not found.
    """
    n = Z.shape[0] + 1  # number of original observations

    # Each cluster is a set of leaf indices it contains
    clusters = {i: {i} for i in range(n)}

    for i, (c1, c2, dist, _) in enumerate(Z):
        new_cluster = clusters[int(c1)] | clusters[int(c2)]
        clusters[n + i] = new_cluster

        if set(leaf_indices).issubset(new_cluster):
            return n + i, dist  # Return cluster ID and its height (y-coordinate)

    return None, None



def fw_typeshuff_wrapper(seed, meta_df, ind_to_id, a_AA_coo):
    '''Wrapper function to recalculate the animal-type vectors 
    We keep a_AA_coo constant (i.e. the food web structure) and shuffle the animal types.
    
    '''
    np.random.seed(seed)
    meta_shuff = meta_df.copy(True)
    type_arr = meta_shuff['type'].to_numpy(copy=True)
    np.random.shuffle(type_arr)
    meta_shuff['type'] = type_arr

    shuff_bpt_AS_coo, animal_row, type_col = bipartite_cooarray(df=meta_shuff, row_col=['node', 'type'], weight=False, row_order=ind_to_id)
    shuff_a_AS_out = (a_AA_coo @ shuff_bpt_AS_coo).tocsr() # out matrix
    shuff_a_AS_in = (a_AA_coo.T @ shuff_bpt_AS_coo).tocsr() # in matrix

    # out matrix normalisation:
    shuff_a_AS_out_normalised = csr_row_norm(shuff_a_AS_out)

    # in matrix normalisation:
    shuff_a_AS_in_normalised = csr_row_norm(shuff_a_AS_in)

    # put together into one dataframe:
    # print(type(type_col[0]))
    all_col_names = np.concatenate([
        np.char.add(type_col, '_out'),
        np.char.add(type_col, '_in')
    ])
    shuff_all_norm_vec_df = pd.DataFrame(sp.sparse.hstack([shuff_a_AS_out_normalised, shuff_a_AS_in_normalised]).toarray(), columns=all_col_names, index=animal_row)
    return shuff_all_norm_vec_df, meta_shuff


def rn_RCshuff_wrapper(seed, a_RI_coo, meta_df, recipe_list, cuisine_col, ingredient_col):
    '''Wrapper function to recalculate the ingredient-space cuisine vectors 
    and the cuisine space ingredient vectors, from keeping Recipe-Ingredient adjacency matrix
    constant. 

    Shuffles the labels in the recipe-cuisine df meta_df.

    '''
    np.random.seed(seed)
    meta_df_bpt_shuff = meta_df.copy(True)
    type_arr = meta_df_bpt_shuff['cuisine'].to_numpy(copy=True)
    np.random.shuffle(type_arr)
    meta_df_bpt_shuff['cuisine'] = type_arr

    a_RC_coo, recipe_row, cuisine_col = bipartite_cooarray( \
        df=meta_df_bpt_shuff.sort_values(['r_id', 'cuisine']), \
        row_col=['r_id', 'cuisine'], \
        weight=False, \
        row_order=list(recipe_list), \
        col_order=list(cuisine_list))

    a_CI_csr = (a_RC_coo.T @ a_RI_coo).tocsr() # csr array

    # normalisation:
    a_CI_csr_normalised = csr_row_norm(a_CI_csr)
    norm_is_c_vec_df = pd.DataFrame(a_CI_csr_normalised.toarray(), columns=ingredient_col, index=cuisine_col)

    a_IC_csr = a_CI_csr.T
    # standarisation - correct for the number of recipes in each cuisine:
    r_sums = np.array(a_RC_coo.tocsr().sum(axis=0)).flatten()  
    r_sums[r_sums == 0] = 1 # divide by 1 instead of zero
    inv_r_sums = sp.sparse.diags(1 / r_sums)
    standardised_a_IC_csr = a_IC_csr @ inv_r_sums

    # normalisation:
    a_IC_csr_s_normalised = csr_row_norm(standardised_a_IC_csr)
    cnorm_cs_i_bpt_df = pd.DataFrame(a_IC_csr_s_normalised.toarray(), columns= cuisine_col, index=ingredient_col)
    return norm_is_c_vec_df, cnorm_cs_i_bpt_df


def rn_RIshuff_wrapper(seed, a_RC_coo, rn_df, recipe_list, cuisine_col, ingredient_list):
    '''Wrapper function to recalculate the ingredient-space cuisine vectors 
    and the cuisine space ingredient vectors, from keeping recipe-cuisine adjacency matrix
    constant. 

    Shuffles the labels in the recipe-ingredient dataframe rn_df 

    '''
    np.random.seed(seed)
    rn_df_shuff = rn_df.copy(True)
    type_arr = rn_df_shuff['r_id'].to_numpy(copy=True)
    np.random.shuffle(type_arr)
    rn_df_shuff['r_id'] = type_arr
    rn_df_shuff.drop_duplicates(inplace=True)

    a_RI_coo, recipe_row, ingredient_col = bipartite_cooarray( \
        df=rn_df_shuff.sort_values(['r_id', 'ingredient']), \
        row_col=['r_id', 'ingredient'], \
        weight=False, \
        row_order=list(recipe_list), \
        col_order=list(ingredient_list))
    # t1 = time.perf_counter()

    a_CI_csr = (a_RC_coo.T @ a_RI_coo).tocsr() # csr array

    # normalisation:
    a_CI_csr_normalised = csr_row_norm(a_CI_csr)
    norm_is_c_vec_df = pd.DataFrame(a_CI_csr_normalised.toarray(), columns=ingredient_col, index=cuisine_col)

    a_IC_csr = a_CI_csr.T
    # standarisation - correct for the number of recipes in each cuisine:
    r_sums = np.array(a_RC_coo.tocsr().sum(axis=0)).flatten()  
    r_sums[r_sums == 0] = 1 # divide by 1 instead of zero
    inv_r_sums = sp.sparse.diags(1 / r_sums)
    standardised_a_IC_csr = a_IC_csr @ inv_r_sums

    # normalisation:
    a_IC_csr_s_normalised = csr_row_norm(standardised_a_IC_csr)
    cnorm_cs_i_bpt_df = pd.DataFrame(a_IC_csr_s_normalised.toarray(), columns= cuisine_col, index=ingredient_col)
    # t2 = time.perf_counter()
    return norm_is_c_vec_df, cnorm_cs_i_bpt_df


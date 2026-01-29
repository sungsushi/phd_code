import os
import pandas as pd



def drosophila_data(datadate='20241119'):
    '''Prepare the drosophila VNC connectome and metadata'''

    dm_datadir = f'../../data/{datadate}_dm_data'
    if not os.path.isdir(dm_datadir):
        os.mkdir(dm_datadir)

    if not os.path.isdir(dm_datadir + '/processed'):
        os.mkdir(dm_datadir + '/processed')

    df = pd.read_csv(f"{dm_datadir}/manc_edges_{datadate}.csv", index_col=0) # edge data frame
    meta_df = pd.read_csv(f"{dm_datadir}/manc_meta_{datadate}.csv", index_col=0) # meta data data frame

    df = df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False).weight.sum()
    df.loc[:, 'bodyId_post'] = df['bodyId_post'].apply(int)
    df.loc[:, 'bodyId_pre'] = df['bodyId_pre'].apply(int)
    meta_df.loc[:, 'bodyId'] = meta_df.bodyId.apply(int)
    return df, meta_df

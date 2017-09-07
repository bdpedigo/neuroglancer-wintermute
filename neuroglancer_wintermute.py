# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 10:46:19 2017

@author: Benjamin Pedigo 
Computational Neuroanatomy Internship 
Allen Institute for Brain Science 
September 2017

Allows the conversion of neuroglancer urls to python dictionaries, and vise 
versa. BASE_URL should be set using an example view and zoom settings that you 
like, the whole thing can be copied and pasted into here. There seem to be some
bugs regarding what zoomFactor is set to. 

PRE should be set to whatever is before the dictionary/json style string in the
URL

Main functions of interest here are url2dict and dict2url. Might be useful if 
transfered over to other code, if necessary. 


"""
import numpy as np
import pandas as pd
import webbrowser

PRE = 'https://neuroglancer-demo.appspot.com/#!'
GRAPH_NAME = r'C:\Users\benjaminp\Desktop\NeuronReconstructions\graph\170605_nt_v11_mst_trimmed_sem_final_edges.csv'
CELL_NAME = r'C:\Users\benjaminp\Desktop\NeuronReconstructions\20170818_cell_list.csv'
BASE_URL = "https://neuroglancer-demo.appspot.com/#!{'layers':{'image':{'type':'image'_'source':'precomputed://gs://neuroglancer/pinky40_v11/image'}_'segmentation':{'type':'segmentation'_'source':'precomputed://gs://neuroglancer/pinky40_v11/watershed_mst_trimmed_sem_remap'_'selectedAlpha':0.24_'segments':['131506448'_'52089750']}}_'navigation':{'pose':{'position':{'voxelSize':[4_4_40]_'voxelCoordinates':[41382_21986_755]}_'orientation':[0.7071067690849304_0.7071067690849304_0_0]}_'zoomFactor':1.8315638888734185}_'perspectiveOrientation':[-0.12623633444309235_-0.9713204503059387_0.19054898619651794_0.06551143527030945]_'perspectiveZoom':60.653065971263395_'showSlices':false_'layout':'xy-3d'}" 

#"https://neuroglancer-demo.appspot.com/#!{'layers':{'image':{'type':'image'_'source':'precomputed://gs://neuroglancer/pinky40_v11/image'}_'segmentation':{'type':'segmentation'_'source':'precomputed://gs://neuroglancer/pinky40_v11/watershed_mst_trimmed_sem_remap'_'selectedAlpha':0.24_'segments':['131506448'_'98408496']}}_'navigation':{'pose':{'position':{'voxelSize':[4_4_40]_'voxelCoordinates':[37692.63671875_27714.02734375_965]}_'orientation':[0.7071067690849304_0.7071067690849304_0_0]}_'zoomFactor':1.8315638888734185}_'perspectiveOrientation':[0.03468929976224899_0.11125264316797256_0.9924432039260864_-0.03839361295104027]_'perspectiveZoom':5459.815003314428_'showSlices':false_'layout':'xy-3d'}"

def get_subgraphs(cell_id, inout):
    '''
    Returns 3 type of subgraphs for each cell
    
    cell_id : int
    inout : str
        'in' or 'out'
    
    Return : 3 pandas dataframes
        (all, smooth, spiny)
    '''
    if inout == 'in':
        all_graph = postsyn_cells_graph[postsyn_cells_graph.segs_2 == cell_id]
        smooth_graph = all_graph[all_graph['segs_1'].isin(smooth_cells)]
        spiny_graph = all_graph[all_graph['segs_1'].isin(spiny_cells)]
    elif inout == 'out':
        all_graph = presyn_cells_graph[presyn_cells_graph.segs_1 == cell_id]
        smooth_graph = all_graph[all_graph['segs_2'].isin(smooth_cells)]
        spiny_graph = all_graph[all_graph['segs_2'].isin(spiny_cells)]
    
    return all_graph, smooth_graph, spiny_graph

def url2dict(url, pre=PRE):
    url = url.replace(pre, '')
    url = url.replace('_', ',')
    url = url.replace('false', 'False')
    dct = eval(url)
    source = dct['layers']['image']['source']
    dct['layers']['image']['source'] = source.replace(',', '_')
    
    source = dct['layers']['segmentation']['source']
    dct['layers']['segmentation']['source'] = source.replace(',', '_')
    
    return dct

def dict2url(dct, pre=PRE):
    url = str(dct)
    url = url.replace(',', '_')
    url = url.replace(' ', '')
    url = url.replace('False', 'false')
    url = pre + url
    
    return url

def get_neighbors(cell_id, adj='all',):
    '''
    Loads the cells that connect to a given cell. Type of connection can be 
    specified using 'adj'
    
    cell_id : int
    adj : str
        In the following options, _(spiny/smooth) refers to connections from or 
        onto a smooth or spiny cell
        'all'
        'in'
        'in_spiny'
        'in_smooth'
        'out'
        'out_spiny'
        'out_smooth'
    '''
    
    if adj == 'all':
        in_all_graph, in_smooth_graph, in_spiny_graph = get_subgraphs(cell_id, 'in')
        in_specific = pd.concat([in_smooth_graph, in_spiny_graph])
        in_neighbors = in_specific.loc[:, 'segs_1'].as_matrix()
        
        out_all_graph, out_smooth_graph, out_spiny_graph = get_subgraphs(cell_id, 'out')
        out_specific = pd.concat([out_smooth_graph, out_spiny_graph])
        out_neighbors = out_specific.loc[:, 'segs_2'].as_matrix()
        neighbors = np.concatenate((in_neighbors, out_neighbors))
        neighbors = np.unique(neighbors)
        neighbors = neighbors.tolist()
        neighbors = [int(n) for n in neighbors]
        return neighbors
    elif adj == 'in':
        in_all_graph, in_smooth_graph, in_spiny_graph = get_subgraphs(cell_id, 'in')
        in_specific = pd.concat([in_smooth_graph, in_spiny_graph])
        in_neighbors = in_specific.loc[:, 'segs_1'].as_matrix()
        neighbors = np.unique(in_neighbors)
        neighbors = neighbors.tolist()
        neighbors = [int(n) for n in neighbors]
        return neighbors

    elif adj == 'out':
        out_all_graph, out_smooth_graph, out_spiny_graph = get_subgraphs(cell_id, 'out')
        out_specific = pd.concat([out_smooth_graph, out_spiny_graph])
        out_neighbors = out_specific.loc[:, 'segs_2'].as_matrix()
        neighbors = np.unique(out_neighbors)
        neighbors = neighbors.tolist()
        neighbors = [int(n) for n in neighbors]
        return neighbors

def load_neighbors(cell, neighbors):
    dct = BASE_DICT
    dct['layers']['segmentation']['segments'] = [cell] + neighbors
    url = dict2url(dct)
    return url

def get_syns(presynaptic, postsynaptic, return_loc=False):
    all_out, _, _ = get_subgraphs(presynaptic, 'out')
    between_graph = all_out[all_out.segs_2 == postsynaptic]
    syns = between_graph.ids.as_matrix().flatten().tolist()
    
    if return_loc:
        rows = full_graph[full_graph.ids.isin(syns)]
        locs = rows.loc[:,'locs_1':'locs_3'].values.tolist()
        for i, loc in enumerate(locs):
            loc = [int(l) for l in loc]
            locs[i] = loc
        return syns, locs
    else:
        return syns
    
    
def go_to(loc, url):
    dct = url2dict(url)
    dct['navigation']['pose']['position']['voxelCoordinates'] = loc
    url = dict2url(dct)
    return url
    
full_graph = pd.read_csv(GRAPH_NAME)

cell_list = pd.read_csv(CELL_NAME)
all_cells = cell_list.as_matrix(columns = ['cell_id'])
all_cells = all_cells.flatten()

cell_type_array = cell_list.as_matrix(columns = ['cell_type'])
cell_type_array = cell_type_array.flatten()

presyn_cells_graph = full_graph[full_graph['segs_1'].isin(all_cells)]
postsyn_cells_graph = full_graph[full_graph['segs_2'].isin(all_cells)]

smooth_cells = all_cells[cell_type_array == 'I']
smooth_cells = smooth_cells.flatten()

spiny_cells = all_cells[cell_type_array == 'E']
spiny_cells = spiny_cells.flatten()

BASE_DICT = url2dict(BASE_URL)
###############################################################################
# example usage


cell = 58045989

print(cell)
# get the list of cells that this one connects ONTO
n = get_neighbors(cell, adj='out')
print(len(n))
# set i to pick a synapse (0 - len(n))
i = 3
# get the url for the two loaded cells
out_neighbors_url = load_neighbors(cell, [n[i]])
# get a list of the synapse ids, a list of the locations
syns_cell, loc_cell = get_syns(cell, n[i], return_loc=True)
# bring the neuroglancer viewer to the first location (ie first synapse)
url = go_to(loc_cell_0[0],  out_neighbors_url)
print(url)

webbrowser.open(url)



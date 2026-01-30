# Node classification in complex networks using local connectivity and annotations

This is a repository of code for analysis and generating the figures for my PhD thesis. 

## How to run:
### Installation:

1. Clone this repo, and download the data folder from Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18422663.svg)](https://doi.org/10.5281/zenodo.18422663)

2. Create the conda environment:
```
conda env create -f environment.yml
conda activate thesis_env
```
3. Then run Jupyter notebooks to do analysis and generate figures into the ```figures``` folder. 



## Contents:
### `01_vectorisation/`
This folder relates to analysis and figures for __Chapter 3: Homologous nodes in annotated networks__

#### `src/00_visualisation.ipynb`
Creates visualisation figure of the recipe network. 

#### `src/01_vectorisation.ipynb`
Performs the connectivity annotation vectorisation and saves files. 

#### `src/02_clustering.ipynb`
Performs the connectivity annotation vector clustering of the Gene Regulatory network, the food web, and the recipe network, and saves all figures. 


### `02_spec_div/`
This folder contains the analysis and figures for __Chapter 4: Diverse and specialised connectivity facilitates multifunctional roles in annotated networks__

#### `src/01_vec_ent_nb`
Performs connectivity vectorisation and specialisation-diversity calculation. Visualises the distributions for _Drosophila_ and _C. elegans_.

#### `src/02_null_models`
Performs null models and verifies statistical signficance. 

#### `src/03_other_specdivs`
Performs specialisation-diversity analysis for other networks. 

### `03_connectome_gm/`
This folder contains the analysis and figures for __Chapter 5: Graph matching methods quantify variability of cell type connectivity in connectomes__

#### `src/01_weg_sizes`
Performs weighted ego graph clustering of three hemilineages across different neighbourhood sizes. 

#### `src/02_ctpurity_stats`
Calculates cluster purity statistics for cell types across different neighbourhood sizes. 

#### `src/02b_nhood_stats`
Calculates neighbourhood size statistics. 

#### `src/03_enriched_dendrogram`
Performs plotting of the dendrogram of the secondary neurons in the VNC, and performs functional enrichment, cell type matching statistics and statistical tests of these matches. 

#### `src/04_vnc_ceval`
Calculates cluster evaluation statistics.

#### `src/07_enriched_cluster_vis`
Visualises enriched clusters with graphviz.

#### `src/08_overlap_stats`
Plots overlap statistics in one figure. 

#### `src/09_gm_stability`
Plots stability figures for graph matching of the same neuron. 

#### `src/10_celegans_gmd`
Graph matching distance of C. elegans neighbourhoods and clustering visualisation. 

### `04_protein_gm/`
This folder contains the analysis and figures for __Chapter 6: Graph matching of secondary structure contact maps reveals hierarchy of conserved folding__

#### `src/01_optimise_params`
Performs simulated annealing to optimise graph matching parameters.

#### `src/02_circular_perm`
Calculates statistics on circular permutation accuracy.

#### `src/03_scop_sample_clustering`
Graph matching distance clustering of a sample of the SCOP data set and visualisation, cluster metric calculations.

#### `src/hpc_runs/`
A folder of the scripts used to run the parallel graph matching distance calculations.


### `05_centrality_vec/`
This folder contains the anlaysis and figures for __Chapter 7: Centrality metrics form interpretable embeddings for complex networks__

#### `src/01_clustering_vis`
Performs centrality vector clustering then visualises them.

#### `src/02_clustering_metrics`
Calculates the clustering metrics.


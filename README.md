# HIN2Vec

*HIN2Vec* learns distributed representations of nodes in heterogeneous information networks (HINs) by capturing the distiguishing metapath relationships between nodes. 
Please refer the paper [here](https://dl.acm.org/citation.cfm?doid=3132847.3132953).

## Prepare

### Compile the source code

    cd model_c/src
    make

## How to use?

### Example

To run *HIN2Vec* on Zachary's karate club network, execute the following command from the repository home directory:<br/>

    python main.py res/karate_club_edges.txt node_vectors.txt metapath_vectors.txt -l 1000 -d 2 -w 2

### Options

See help for the other available options to use with *HIN2Vec*.<br/>

    python main.py --help

### Input HIN format

The supported input format is an edgelist (separated by TAB):

    node1_name node2_type node2_name node2_type edge_type
                    
The input graph is assumed to be directed by default, which means that for an edge in a undirected graph, you need to add two directed edges. For example:

   1   U   11  U   U-U 
   ...
   11   U   1  U   U-U 
   ...

### Output

After learning, HIN2Vec outputs two files: node representations and metapath representations.

The node representation file has *n+1* lines for a graph with *n* nodes with the following format. 

    num_of_nodes dim_of_representation
    node_id dim1 dim2 ... dimd
    ...

where dim1, ... , dimd is the *d*-dimensional node representation learned by *HIN2Vec*.

The metapath representation file has *k+1* lines for a graph with *k* targeted metapath relationships with the following format. 

    num_of_metapaths dim_of_representation
    metapath1 dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional metapath representation learned by *HIN2Vec*. The number of target metapaths is related to the window size set for learning and the schema of the given graph.

### Check the learned vectors

Members in the Zachary's karate club network natually seperate into two groups. To treat the group classification for members as a binary classification:<br/>

    python tools/exp_classification.py node_vectors.txt res/karate_club_groups.txt

## Citing

If you find *HIN2Vec* useful for your research, please cite the following paper:

    @inproceedings{fu2017hin2vec,
        title={HIN2Vec: Explore Meta-paths in Heterogeneous Information Networks for Representation Learning},
        author={Fu, Tao-yang and Lee, Wang-Chien and Lei, Zhen},
        booktitle={Proceedings of the 2017 ACM on Conference on Information and Knowledge Management},
        pages={1797--1806},
        year={2017},
        organization={ACM}
    }


### Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <txf225@cse.psu.edu> or <csiegoat@gmail.com>.

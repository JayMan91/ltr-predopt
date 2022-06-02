### Instruction to download the data
For this experiment, we use the cora dataset used in the paper ["Melding the data-decisions pipeline: Decision-focused learning for combinatorial optimization."](https://doi.org/10.1609/aaai.v33i01.33011658) by Bryan Wilder, Bistra Dilkina and Milind Tambe.
- Run `get_data.sh` to download preprocessed CORA files
- Run `make_cora_dataset.py` to build bipartite graphs and feature vectors associated with each node.

### Reproducing the results
In order to reproduce the result, run the files, whose name start with `test` and use the hyperparameter as stated in **Appendix B**.
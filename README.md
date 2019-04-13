# Pytorch RandWire
Pytorch implementation of RandWire network in [Exploring Randomly Wired Neural Networks for Image Recognition](https://arxiv.org/abs/1904.01569) by Xie et al. (2019)

### Requirements
- [PyTorch](https://pytorch.org/)
- [NetworkX](https://networkx.github.io/)
- [tqdm](https://github.com/tqdm/tqdm)
- [TensorFlow](https://www.tensorflow.org/) for logging only.

### Features
- Simple and readable random network generation and usage using NetworkX package.
- Random sub-network generated from DAG (directed acyclic graph) uses simple live variable analysis to reduce dynamic memory usage. (64 images per GPU requires less than 8 GB of memory)

### TODOs
- Train on ImageNet and record statistics.
- Create graph visualization tool for fancy looking ;)
- Build RandWireTiny module running CIFAR-10 for lighter demo.
- Build computational cost calculation method that returns #FLOPs.
- Use compiler optimization technique to further reduce memory usage in the forward paths of random networks.
- Use topological sort algorithm other than NetworkX's for parallel computation of uncorrelated nodes.
- Update README

### Author
Jaerin Lee
<br/>Seoul National University
<br/>ironjr@snu.ac.kr

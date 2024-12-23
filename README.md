# PersonalBPNet

A small modification to bpnetlite's BPNet to accomodate large validation datasets.

Redid the validation loop to work with a PyTorch DataLoader, rather than having to load the whole validation set into memory at once. Also, the model checkpoints save the optimizer state dict and epoch number in addition to the model state dict, so that training can be resumed from a checkpoint.

Additionally, we include clipnet_pytorch, which implements a BPNet-like NN with added batch norm and maxpool layers, similar to what was done with CLIPNET in tensorflow.

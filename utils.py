import numpy as np
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

def obtain_idx_random(seed,node_idxs, size):
    ### current random to implement
    size = min(len(node_idxs),size)
    rs = np.random.RandomState(seed)
    choice = rs.choice(len(node_idxs),size)
    return node_idxs[choice]

    
def obtain_idx_distance(args, node_idxs, size, model, dataset, device):
    size = min(len(node_idxs),size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model.pretrain(dataloader, args, verbose=args.debug)
    if(args.normal_selection == 'distance'):
        # obtain the repreentations of all graphs (or )
        gs = torch.FloatTensor([]).to(device)
        for data in dataloader:
            data = data.to(device)
            _, g = model.encoder(data.x, data.edge_index, data.batch)
            gs = torch.cat((gs,g),dim=0)
        # distance_matrix = F.pairwise_distance(gs[0],gs,keepdim=True)
        gs = gs.detach().cpu().numpy()
        distance_matrix = pairwise_distances(gs)
        sum_distances = (distance_matrix).sum(axis=0)
        idx_select = np.argsort(sum_distances)[::-1][:size]
        return idx_select
    elif(args.normal_selection == 'cluster'):
        # use K-means to cluster into n groups, and find the nodes with the largest distance to its cluster center
        ## obtain the repreentations of all graphs (or )
        gs = torch.FloatTensor([]).to(device)
        labels = torch.LongTensor([]).to(device)
        for data in dataloader:
            data = data.to(device)
            _, g = model.encoder(data.x, data.edge_index, data.batch)
            gs = torch.cat((gs,g),dim=0)
            labels = torch.cat((labels,data.y),dim=0)
        labels = labels.detach().cpu().numpy()
        ## distance_matrix = F.pairwise_distance(gs[0],gs,keepdim=True)
        gs = gs.detach().cpu().numpy()
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=1)
        kmeans.fit(gs)
        y_pred = kmeans.predict(gs)
        cluster_centers = kmeans.cluster_centers_
        distances = []
        for idx in node_idxs:
            tmp_center_label = y_pred[idx]
            tmp_center_g = cluster_centers[tmp_center_label]
            tmp_g = gs[idx]
            # tmp_dist = pairwise_distances(tmp_g,tmp_center_g).mean()
            tmp_dist = np.linalg.norm(tmp_g - tmp_center_g)
            distances.append(tmp_dist)
        distances = np.array(distances)
        idx_select = np.argsort(distances)[::-1][:size]
        return idx_select
    
def get_split_self(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8, seed: int = 42, device=None):
    """Return indices for train, test, and valid splits."""
    assert train_ratio + test_ratio <= 1
    rs = np.random.RandomState(seed)
    perm = rs.permutation(num_samples)
    indices = torch.tensor(perm).to(device)
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    idx_test = indices[train_size: test_size + train_size]
    # randomly split idx_test
    idx_clean_test = idx_test[:int(len(idx_test)/2)]
    idx_atk = idx_test[int(len(idx_test)/2):]
    # indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'test': indices[train_size: test_size + train_size],
        'valid': indices[test_size + train_size:],
        'clean_test': idx_test[:int(len(idx_test)/2)],
        'atk': idx_test[int(len(idx_test)/2):]
    }


def accuracy(output, labels):
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def parameter_diff(w1,w2):

    return torch.norm(torch.nn.utils.parameters_to_vector(w1) - torch.nn.utils.parameters_to_vector(w2), p=2)


def clone_module(module, memo=None):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().

    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.

    **Arguments**

    * **module** (Module) - Module to be cloned.

    **Return**

    * (Module) - The cloned module.

    **Example**

    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[buff_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone



def update_module(module, updates=None, memo=None):
    r"""
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Updates the parameters of a module in-place, in a way that preserves differentiability.

    The parameters of the module are swapped with their update values, according to:
    \[
    p \gets p + u,
    \]
    where \(p\) is the parameter, and \(u\) is its corresponding update.


    **Arguments**

    * **module** (Module) - The module to update.
    * **updates** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the tensors in .update attributes.

    **Example**
    ~~~python
    error = loss(model(X), y)
    grads = torch.autograd.grad(
        error,
        model.parameters(),
        create_graph=True,
    )
    updates = [-lr * g for g in grads]
    l2l.update_module(model, updates=updates)
    ~~~
    """
    if memo is None:
        memo = {}
    if updates is not None:
        params = list(module.parameters())
        if not len(updates) == len(list(params)):
            msg = 'WARNING:update_module(): Parameters and updates have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(updates)) + ')'
            print(msg)
        for p, g in zip(params, updates):
            p.update = g

    # Update the params
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p in memo:
            module._parameters[param_key] = memo[p]
        else:
            if p is not None and hasattr(p, 'update') and p.update is not None:
                updated = p + p.update
                p.update = None
                memo[p] = updated
                module._parameters[param_key] = updated

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff in memo:
            module._buffers[buffer_key] = memo[buff]
        else:
            if buff is not None and hasattr(buff, 'update') and buff.update is not None:
                updated = buff + buff.update
                buff.update = None
                memo[buff] = updated
                module._buffers[buffer_key] = updated

    # Then, recurse for each submodule
    for module_key in module._modules:
        module._modules[module_key] = update_module(
            module._modules[module_key],
            updates=None,
            memo=memo,
        )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(module, 'flatten_parameters'):
        module._apply(lambda x: x)
    return module

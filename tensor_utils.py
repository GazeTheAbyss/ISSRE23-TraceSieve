from typing import *

import dgl
import numpy as np
import torch
from tensorkit import tensor as T
from torch._C import device

from tracegnn.models.vgae.constants import *
from tracegnn.models.vgae.types import *

__all__ = [
    'latency_onehot_to_mask',
    'edge_logits_by_dot_product',
    'dense_to_triu',
    'triu_to_dense',
    'dense_triu_adj',
    'pad_node_feature',
    'get_moments',
    'node_count_mask',
]


def latency_onehot_to_mask(onehot: T.Tensor) -> T.Tensor:
    """
    >>> onehot = T.as_tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> T.to_numpy(latency_onehot_to_mask(onehot))
    array([[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]])
    >>> T.to_numpy(latency_onehot_to_mask(T.cast(onehot, dtype=T.float32)))
    array([[1., 0., 0.],
           [1., 1., 0.],
           [1., 1., 1.]], dtype=float32)
    """
    origin_dtype = T.get_dtype(onehot)
    onehot = T.as_tensor(onehot, dtype=T.boolean)
    shape = T.shape(onehot)
    right = shape[-1] - 1
    mask = T.full(shape, False, dtype=T.boolean)
    mask[..., right] = onehot[..., right]
    while right > 0:
        old_right = right
        right -= 1
        mask[..., right] = T.logical_or(mask[..., old_right], onehot[..., right])
    return T.cast(mask, dtype=origin_dtype)


def edge_logits_by_dot_product(h: T.Tensor) -> T.Tensor:
    left = h
    right = T.swap_axes(h, -1, -2)
    return T.matmul(left, right)


def triu_mask(node_count: int) -> T.Tensor:
    return torch.triu(T.full([node_count, node_count], True, T.boolean), 1)


def batch_edge_mask(g: dgl.DGLGraph) -> T.Tensor:
    i = 0
    mask = T.zeros([g.num_nodes(), g.num_nodes()], dtype=T.boolean)

    for cnt in list(g.batch_num_nodes()):
        mask[i:i+cnt, i:i+cnt] = True
        i += cnt
    
    return mask


def dense_to_triu(x: T.Tensor, node_count: int) -> T.Tensor:
    mask = triu_mask(node_count)
    shape = T.shape(x)
    return T.reshape(x, shape[:-2] + [-1])[..., mask.reshape(-1)]


def triu_to_dense(x: T.Tensor,
                  node_count: int,
                  pad_value: Union[int, float] = 0) -> T.Tensor:
    mask = triu_mask(node_count).reshape(-1)
    ret = T.full([node_count * node_count], pad_value, dtype=T.get_dtype(x))
    ret[mask] = x
    return T.reshape(ret, [node_count, node_count])


def dense_triu_adj(g: dgl.DGLGraph, node_count: int, reverse: bool = False) -> T.Tensor:
    adj = T.zeros([node_count, node_count], dtype=T.float32)
    adj[:g.num_nodes(), :g.num_nodes()] = g.adj(reverse, ctx=g.device).to_dense()
    # u, v = g.edges()
    # if reverse:
    #     v, u = u, v
    # adj[u, v] = 1
    # adj = to_dense_adj(
    #     T.stack([u, v], axis=0),
    #     max_num_nodes=node_count
    # )
    return dense_to_triu(adj, node_count)


def pad_node_feature(G: TraceGraphBatch,
                     feature_name: str,
                     max_node_count: int = MAX_NODE_COUNT):
    # inspect graph count
    graph_count = len(G.dgl_graphs)

    # inspect features
    vec = G.dgl_batch.ndata[feature_name]
    value_shape = T.shape(vec)[1:]
    dtype = T.get_dtype(vec)
    device = T.get_device(vec)

    # todo: whether or not it's better to use concat instead of copying into a new tensor?
    with T.no_grad():
        ret = T.zeros(
            [graph_count, max_node_count] + value_shape,
            dtype=dtype,
            device=device,
        )
        for i in range(graph_count):
            vec = G.dgl_graphs[i].ndata[feature_name]
            ret[i, :T.shape(vec)[0]] = vec
    return ret


def get_moments(x,
                axis: Optional[List[int]] = None,
                clip_var: bool = False,
                ) -> Tuple[T.Tensor, T.Tensor]:
    mean = T.reduce_mean(x, axis=axis)
    var = T.reduce_mean(x ** 2, axis=axis) - mean ** 2
    if clip_var:
        var = T.maximum(var, dtype=T.get_dtype(var))
    return mean, var


def node_count_mask(node_count,
                    max_node_count: int,
                    dtype: Optional[str] = None) -> T.Tensor:
    h = T.arange(0, max_node_count, dtype=T.get_dtype(node_count))
    node_count = T.expand_dim(node_count, axis=-1)
    h = h < node_count
    if dtype is not None:
        h = T.cast(h, dtype)
    return h

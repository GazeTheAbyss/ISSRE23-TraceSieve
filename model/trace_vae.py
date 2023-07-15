from typing import *

import mltk
import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.typing_ import TensorOrData

from ..constants import *
from ..tensor_utils import *
from ..types import *
from .operation_embedding import *
from .vae_model import *

__all__ = [
    'TraceVAEConfig',
    'TraceVAE',
]


class TraceVAEConfig(mltk.Config):
    operation_embedding_dim: int = 40
    use_latency: bool = True

    struct: TraceStructVAEConfig = TraceStructVAEConfig()
    latency: TraceLatencyVAEConfig = TraceLatencyVAEConfig()


class TraceVAE(tk.layers.BaseLayer):

    config: TraceVAEConfig
    num_operations: int

    def __init__(self, config: TraceVAEConfig, num_operations: int):
        super().__init__()

        # ===================
        # memorize the config
        # ===================
        self.config = config
        self.num_operations = num_operations

        # ==============
        # the components
        # ==============
        self.operation_embedding = OperationEmbedding(
            num_operations=num_operations,
            embedding_dim=config.operation_embedding_dim,
        )
        self.vae_model = TraceVAEModel(config.struct, config.latency, self.operation_embedding)

    def _is_attr_included_in_repr(self, attr: str, value: Any) -> bool:
        if attr == 'config':
            return False
        return super()._is_attr_included_in_repr(attr, value)

    def _call_graph_batch_build(self, G: TraceGraphBatch):
        G.build_dgl(
            add_self_loop=True,
            directed=False,
            # directed=('reverse' if self.config.edge.reverse_directed else False),
        )

    def q(self,
          G: TraceGraphBatch,
          observed: Optional[Mapping[str, TensorOrData]] = None,
          n_z: Optional[int] = None,
          ):
        self._call_graph_batch_build(G)
        net = tk.BayesianNet(observed=observed)

        self.vae_model.q(net, G.dgl_batch, n_z=n_z)

        return net

    def p(self,
          observed: Optional[Mapping[str, TensorOrData]] = None,
          G: Optional[TraceGraphBatch] = None,  # the observed `G`
          n_z: Optional[int] = None,
          use_biased: bool = False,
          latency_logstd_min: Optional[float] = None,
          std_result_dict: Dict[int, List[float]] = None,
          std_max_limit: T.Tensor = None
          ) -> tk.BayesianNet:
        config = self.config

        # populate `observed` from `G` if specified, and construct net
        if G is not None:
            self._call_graph_batch_build(G)
            g = G.dgl_batch
            observed = observed or {}

            # struct
            observed['adj'] = g.adj(ctx=g.device).to_dense()
            observed['node_type'] = g.ndata['node_type']

            # latency
            observed['latency'] = g.ndata['latency'][..., :LATENCY_DIM]
        else:
            g = None

        # the Bayesian net
        net = tk.BayesianNet(observed=observed)

        # call components
        node_type_tensor, logstd_tensor = self.vae_model.p(net, g, n_z=n_z, use_biased=use_biased,
                                                           latency_logstd_min=latency_logstd_min, std_max_limit=std_max_limit)

        # Append to std_result_dict
        if std_result_dict is not None:
            for i in range(node_type_tensor.size(0)):
                cur_type = node_type_tensor[i].item()

                if cur_type not in std_result_dict:
                    std_result_dict[cur_type] = []

                std_result_dict[cur_type].append(logstd_tensor[i].item())

        return net

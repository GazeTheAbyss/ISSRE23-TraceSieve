from typing import *

import dgl
import mltk
import tensorkit as tk
import torch
from tensorkit import tensor as T
from ..constants import *
from ..distributions import *
from ..tensor_utils import *
from ..tensor_utils import batch_edge_mask
from .gnn_layers import *
from .model_utils import *
from .operation_embedding import *
from .realnvp_flow import *

import torch.nn.functional as F

__all__ = [
    'TraceStructVAEConfig',
    'TraceLatencyVAEConfig',
    'TraceVAEModel',
]


class TraceLatencyVAEConfig(mltk.Config):
    # whether to use the operation embedding? (but grad will be blocked)
    use_operation_embedding: bool = True

    # the dimension of z2 (to encode latency)
    z2_dim: int = 10

    # the config of posterior / prior flow
    realnvp: RealNVPFlowConfig = RealNVPFlowConfig()

    # whether to use BatchNorm?
    use_batch_norm: bool = True

    class encoder(mltk.Config):
        # ================
        # h(G) for q(z2|G)
        # ================
        # the gnn layer config
        gnn: GNNLayerConfig = GNNLayerConfig()

        # the gnn layer sizes for q(z2|...)
        gnn_layers: List[int] = [500, 500, 500, 500]

        # whether to stop gradient to operation_embedding along this path?
        operation_embedding_stop_grad: bool = True

        # =======
        # q(z2|G)
        # =======
        # the minimum of logstd
        z2_logstd_min: Optional[float] = -9

        # whether to use realnvp posterior flow?
        use_posterior_flow: bool = False

    class decoder(mltk.Config):
        # ====================
        # decoder architecture
        # ====================
        use_prior_flow: bool = False

        # whether to use `z2` directly as context, instead of passing through
        # the graph embedding layers?
        z2_as_context: bool = True

        # =======
        # latency
        # =======
        # gnn layer config
        gnn: GNNLayerConfig = GNNLayerConfig()

        # the node types from node embedding e
        gnn_layers: List[int] = [500, 500, 500, 500]

        # hidden layers for graph embedding from z
        graph_embedding_layers: List[int] = [500, 500]

        # size of the latent embedding e
        latent_embedding_size: int = 40

        # whether to stop gradient to operation_embedding along this path?
        operation_embedding_stop_grad: bool = True

        # ==============
        # p(latency|...)
        # ==============
        # the minimum value for latency logstd
        latency_logstd_min: Optional[float] = -7

        # whether to use mask on p(latency|...)?
        use_latency_mask: bool = True

        # whether to clip the latency to one dim even if three dim is provided?
        clip_latency_to_one_dim: bool = False

        # whether to use biased in p(latency|...)?
        use_biased_latency: bool = True

        # whether to use `AnomalyDetectionNormal`?
        use_anomaly_detection_normal: bool = False

        # the `std_threshold` for AnomalyDetectionNormal or BiasedNormal in testing
        biased_normal_std_threshold: float = 4.0

        # the `std_threshold` for SafeNormal in training
        safe_normal_std_threshold: float = 6.0


class TraceStructVAEConfig(mltk.Config):
    # the dimension of z (to encode adj & node_type)
    z_dim: int = 3

    # the config of posterior / prior flow
    realnvp: RealNVPFlowConfig = RealNVPFlowConfig()

    # whether to use BatchNorm?
    use_batch_norm: bool = True

    class encoder(mltk.Config):
        # ===============
        # h(G) for q(z|G)
        # ===============
        # the gnn layer config
        gnn: GNNLayerConfig = GNNLayerConfig()

        # the gnn layer sizes for q(z|...)
        gnn_layers: List[int] = [250, 250, 250, 250]

        # ======
        # q(z|G)
        # ======
        # the minimum of logstd
        z_logstd_min: Optional[float] = -9


    class decoder(mltk.Config):
        # =========
        # structure
        # =========
        # gnn layer config
        gnn: GNNLayerConfig = GNNLayerConfig()

        # the node types from node embedding e
        gnn_layers: List[int] = [250, 250, 250, 250]

        # hidden layers for p(node_count|z)
        node_count_layers: List[int] = [250]

        # hidden layers for graph embedding from z
        graph_embedding_layers: List[int] = [500, 500]



class TraceVAEModel(tk.layers.BaseLayer):

    struct_config: TraceStructVAEConfig
    num_operations: int

    def __init__(self,
                 struct_config: TraceStructVAEConfig,
                 latency_config: TraceLatencyVAEConfig,
                 operation_embedding: OperationEmbedding,
                 ):
        super().__init__()

        # ===================
        # memorize the config
        # ===================
        self.struct_config = struct_config
        self.latency_config = latency_config

        # =============================
        # node embedding for operations
        # =============================
        self.operation_embedding = operation_embedding
        self.num_operations = operation_embedding.num_operations

        # ========================
        # standard layer arguments
        # ========================
        layer_args = tk.layers.LayerArgs()
        layer_args.set_args(['dense'], activation=tk.layers.LeakyReLU)
        if struct_config.use_batch_norm:
            layer_args.set_args(['dense'], normalizer=tk.layers.BatchNorm)

        # ==================
        # q(z|adj,node_type)
        # ==================
        output_size, gnn_layers = make_gnn_layers(
            struct_config.encoder.gnn,
            self.operation_embedding.embedding_dim+LATENCY_DIM,
            struct_config.encoder.gnn_layers,
        )
        self.qz_gnn_layers = GNNSequential(gnn_layers)

        self.qz_mean = tk.layers.Linear(output_size, struct_config.z_dim)
        self.qz_logstd = tk.layers.Linear(output_size, struct_config.z_dim)

        # note: p(adj) = outer-dot(e)

        # ==================
        # p(node_type|z)
        # ==================
        input_size = struct_config.z_dim

        output_size, gnn_layers = make_gnn_layers(
            struct_config.decoder.gnn,
            input_size,
            struct_config.decoder.gnn_layers,
        )
        self.feature_decoder = GNNSequential(
            gnn_layers +
            [
                GraphConv(output_size, self.num_operations+LATENCY_DIM*2),  # p(node_type, latency|z)
            ]
        )

    def _is_attr_included_in_repr(self, attr: str, value: Any) -> bool:
        if attr == 'config':
            return False
        return super()._is_attr_included_in_repr(attr, value)

    def q(self,
          net: tk.BayesianNet,
          g: dgl.DGLGraph,
          n_z: Optional[int] = None):
        config = self.struct_config

        # embedding lookup
        h = self.operation_embedding(g.ndata['node_type'])
        
        # latency embedding
        h = T.concat([h, g.ndata['latency'][..., :LATENCY_DIM]], axis=-1)

        # feed into gnn and get node embeddings
        h = self.qz_gnn_layers(g, h)

        # mean and logstd for q(z|G)
        z_mean = self.qz_mean(h)
        z_logstd = T.maybe_clip(
            self.qz_logstd(h),
            min_val=config.encoder.z_logstd_min,
        )

        # add 'z' random variable
        # z: [bn x d]
        qz = tk.Normal(mean=z_mean, logstd=z_logstd, event_ndims=1)
        z = net.add('z', qz, n_samples=n_z)

    def p(self,
          net: tk.BayesianNet,
          g: dgl.DGLGraph,
          n_z: Optional[int] = None,
          use_biased: bool = False,
          latency_logstd_min: Optional[float] = None,
          std_max_limit: T.Tensor = None):
        # sample z ~ p(z)
        pz = tk.UnitNormal([g.num_nodes(), self.struct_config.z_dim], event_ndims=1)
        z = net.add('z', pz, n_samples=n_z)

        # p(A|e)
        edge_logits = edge_logits_by_dot_product(z.tensor)
        edge_mask = batch_edge_mask(g)
    
        if len(edge_logits.shape) > 2:
            edge_mask = edge_mask.unsqueeze(0).expand_as(edge_logits)

        # edge_logits = dense_to_triu(edge_logits, MAX_NODE_COUNT)

        if use_biased:
            p_adj = BiasedBernoulli(
                alpha=MAX_NODE_COUNT,
                threshold=0.5,
                logits=edge_logits,
                event_ndims=0
            )
        else:
            p_adj = tk.Bernoulli(logits=edge_logits)

        # Add mask for adj
        p_adj = MaskedDistribution(
            distribution=p_adj,
            mask=edge_mask,
            event_ndims=1
        )

        adj = net.add('adj', p_adj, n_samples=n_z)

        decoded_features = self.feature_decoder(g, z.tensor)

        # Node type
        node_type_logits = decoded_features[..., :self.num_operations]
        p_node_type = tk.Categorical(logits=node_type_logits, event_ndims=0)
        node_type = net.add('node_type', p_node_type)

        # mean & logstd for p(latency|z2,G)
        if latency_logstd_min is not None:
            if self.latency_config.decoder.latency_logstd_min is not None:
                latency_logstd_min = max(
                    latency_logstd_min,
                    self.latency_config.decoder.latency_logstd_min
                )
        else:
            latency_logstd_min = self.latency_config.decoder.latency_logstd_min

        # Latency
        latency_mean = decoded_features[..., self.num_operations:self.num_operations+1]
        latency_logstd = T.maybe_clip(
            decoded_features[..., self.num_operations+1:self.num_operations+2],
            min_val=latency_logstd_min
        )

        # Apply max limit
        if std_max_limit is not None:
            node_type_one_hot = T.one_hot(
                node_type.tensor, n_classes=std_max_limit.size(0))
            std_limit_value = torch.sum(
                node_type_one_hot * std_max_limit, dim=-1)

            latency_logstd = torch.min(
                latency_logstd, std_limit_value.unsqueeze(-1))

        if self.training:
            p_latency = SafeNormal(
                std_threshold=self.latency_config.decoder.safe_normal_std_threshold,
                mean=latency_mean,
                logstd=latency_logstd,
                event_ndims=1
            )
        else:
            if use_biased:
                p_latency = BiasedNormal(
                        alpha=MAX_NODE_COUNT,
                        std_threshold=4.0,
                        mean=latency_mean,
                        logstd=latency_logstd,
                        event_ndims=1,
                    )
            else:
                p_latency = tk.Normal(
                    mean=latency_mean,
                    logstd=latency_logstd,
                    event_ndims=1
                )

        latency = net.add('latency', p_latency, n_samples=n_z)

        return node_type.tensor, latency_logstd

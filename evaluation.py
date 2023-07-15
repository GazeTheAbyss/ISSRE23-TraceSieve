import json
import math
import traceback
from pprint import pprint
from typing import *
import torch

import mltk
import seaborn as sns
import tensorkit as tk
import yaml
from tensorkit import tensor as T
from tqdm import tqdm

from tracegnn.utils import analyze_anomaly_nll
from tracegnn.visualize import *
from ...data import TraceGraph, TraceGraphNode
from ...utils import TraceGraphLatencyRangeFile
from .graph_utils import p_net_to_trace_graphs
from .model import TraceVAE
from .types import TraceGraphBatch

__all__ = [
    'do_evaluate_nll',
    'do_evaluate_prior',
]


def do_evaluate_nll(test_stream: mltk.DataStream,
                    vae: TraceVAE,
                    id_manager: TraceGraphIDManager,
                    latency_range: TraceGraphLatencyRangeFile,
                    n_z: int,
                    use_biased: bool = True,
                    no_latency: bool = False,
                    latency_logstd_min: Optional[float] = None,
                    test_loop=None,
                    summary_writer=None,
                    clip_nll=None,
                    use_embeddings: bool = False,
                    num_embedding_samples=None,
                    nll_output_file=None,
                    proba_cdf_file=None,
                    auc_curve_file=None,
                    latency_hist_file=None,
                    dataset_name=None,
                    save_filename=None,
                    std_max_limit=None
                    ):
    # result buffer
    nll_list = []
    label_list = []
    z_buffer = []  # the z embedding buffer of the graph
    z2_buffer = []  # the z2 embedding buffer of the graph
    z_label = []  # the label for z and z2
    latency_samples = {}

    def add_embedding(buffer, label, tag, limit=None):
        if limit is not None:
            indices = np.arange(len(buffer))
            np.random.shuffle(indices)
            indices = indices[:limit]
            buffer = buffer[indices]
            label = label[indices]
        summary_writer.add_embedding(
            buffer,
            metadata=label,
            tag=tag,
        )

    # run evaluation
    def eval_step(trace_graphs):
        G = TraceGraphBatch(
            id_manager=id_manager,
            latency_range=latency_range,
            trace_graphs=trace_graphs,
        )
        chain = vae.q(G, n_z=n_z).chain(
            vae.p,
            G=G,
            n_z=n_z,
            use_biased=use_biased,
            latency_logstd_min=latency_logstd_min,
            latent_axis=0,
            std_max_limit=std_max_limit
        )
        loss = chain.vi.training.sgvb(reduction='none')
        nll = -chain.vi.evaluation.is_loglikelihood()

        loss_temp, nll_temp = [], []

        index = 0
        for cnt in list(G.dgl_batch.batch_num_nodes()):
            loss_temp.append(torch.mean(loss[index:index+cnt]).item())
            nll_temp.append(torch.mean(nll[index:index+cnt]).item())
            index += cnt

        loss = torch.tensor(loss_temp, device=loss.device)
        nll = torch.tensor(nll_temp, device=nll.device)

        # clip the nll, and treat 'NaN' or 'Inf' nlls as `config.test.clip_nll`
        if clip_nll is not None:
            clip_limit = T.float_scalar(clip_nll)
            loss = T.where(loss < clip_limit, loss, clip_limit)
            nll = T.where(nll < clip_limit, nll, clip_limit)

        # the nlls and labels of this step
        step_label_list = np.array([
            0 if not g.data.get('is_anomaly') else (
                1 if g.data['anomaly_type'] == 'drop' else 2)
            for g in trace_graphs
        ])

        # # inspect the internals of every trace graph
        # if 'latency' in chain.p:
        #     p_latency = chain.p['latency'].distribution.base_distribution
        #     p_latency_mu, p_latency_std = p_latency.mean, p_latency.std
        #     if len(T.shape(p_latency.mean)) == 4:
        #         p_latency_mu = p_latency_mu[0]
        #         p_latency_std = p_latency_std[0]

        #     latency_sample = T.to_numpy(T.random.normal(p_latency_mu, p_latency_std))

        #     for i, tg in enumerate(trace_graphs):
        #         assert isinstance(tg, TraceGraph)
        #         if step_label_list[i] == 0:
        #             for j in range(tg.node_count):
        #                 node_type = int(T.to_numpy(G.dgl_graphs[i].ndata['node_type'][j]))
        #                 if node_type not in latency_samples:
        #                     latency_samples[node_type] = []
        #                 mu, std = latency_range[node_type]
        #                 latency_samples[node_type].append(latency_sample[i, j, 0] * std + mu)

        # if use_embeddings:
        #     for i in range(len(trace_graphs)):
        #         if step_label_list[i] == 0:
        #             node_type = trace_graphs[i].root.operation_id
        #             node_label = id_manager.operation_id.reverse_map(node_type)
        #             z_label.append(node_label)
        #             z_buffer.append(T.to_numpy(chain.q['z'].tensor[0, i]))
        #             if 'z2' in chain.q:
        #                 z2_buffer.append(T.to_numpy(chain.q['z2'].tensor[0, i]))

        # memorize the outputs
        nll_list.extend(T.to_numpy(nll))
        label_list.extend(step_label_list)

        # return a dict of the test result
        ret = {}
        normal_losses = T.to_numpy(loss)[step_label_list == 0]
        if len(normal_losses) > 0:
            test_loss = np.nanmean(normal_losses)
            if not math.isnan(test_loss):
                ret['loss'] = test_loss
        return ret

    with tk.layers.scoped_eval_mode(vae), T.no_grad():
        # run test on test set
        if test_loop is not None:
            with test_loop.timeit('eval_time'):
                test_loop.run(eval_step, test_stream)
        else:
            for [trace_graphs] in tqdm(test_stream, total=test_stream.batch_count, desc='Evaluation'):
                eval_step(trace_graphs)

        # save the evaluation results
        nll_list = np.asarray(nll_list)
        label_list = np.asarray(label_list)
        if nll_output_file is not None:
            np.savez_compressed(
                nll_output_file,
                nll_list=nll_list,
                label_list=label_list,
            )

        # analyze nll
        if std_max_limit is None:
            method_name = 'vgae'
        else:
            method_name = 'vgae-std-limit'

        result_dict = analyze_anomaly_nll(
            nll_list=nll_list,
            label_list=label_list,
            proba_cdf_file=proba_cdf_file,
            auc_curve_file=auc_curve_file,
            save_dict=True,
            method=method_name,
            dataset=dataset_name,
            save_filename=save_filename
        )

        # # latency hist
        # if latency_hist_file is not None and latency_samples:
        #     fig = plot_latency_hist(
        #         latency_samples,
        #         id_manager=id_manager,
        #     )
        #     if latency_hist_file == ':show:':
        #         plt.show()
        #     elif latency_hist_file:
        #         plt.savefig(latency_hist_file)
        #     plt.close()

        # # z embedding
        # if use_embeddings:
        #     # add the operation embedding
        #     operation_buffer = T.to_numpy(vae.operation_embedding(
        #         T.arange(0, id_manager.num_operations, dtype=T.int64)))
        #     operation_label = [
        #         id_manager.operation_id.reverse_map(i)
        #         for i in range(id_manager.num_operations)
        #     ]
        #     add_embedding(operation_buffer, operation_label, 'operation')

        #     # add z & z2 embedding
        #     z_label = np.stack(z_label, axis=0)
        #     add_embedding(
        #         np.stack(z_buffer, axis=0),
        #         z_label,
        #         tag='z',
        #         limit=num_embedding_samples
        #     )
        #     if z2_buffer:
        #         add_embedding(
        #             np.stack(z2_buffer, axis=0),
        #             z_label,
        #             tag='z2',
        #             limit=num_embedding_samples
        #         )

    # return the results
    return result_dict


def do_evaluate_prior(vae: TraceVAE,
                      id_manager: TraceGraphIDManager,
                      latency_range: TraceGraphLatencyRangeFile,
                      n_samples: int,
                      batch_size: int,
                      eval_n_z: int,
                      nll_threshold: Optional[float] = None,
                      use_biased: bool = True,
                      output_file: Optional[str] = None,
                      latency_hist_out: Optional[str] = None,
                      ):
    with tk.layers.scoped_eval_mode(vae), T.no_grad():
        # results
        sample_count = 0
        drop_count = 0
        result_dict = {}
        latency_map = {}

        def add_sample(g: TraceGraph):
            if latency_hist_out is not None:
                for _, nd in g.iter_bfs():
                    assert isinstance(nd, TraceGraphNode)
                    if nd.operation_id not in latency_map:
                        latency_map[nd.operation_id] = []
                    latency_map[nd.operation_id].append(nd.features.avg_latency)

        # run by sample from prior
        n_batches = (n_samples + batch_size - 1) // batch_size
        for _ in tqdm(range(n_batches), total=n_batches, desc='Sample graphs from prior'):
            # sample from prior
            p = vae.p(n_z=batch_size)
            trace_graphs = p_net_to_trace_graphs(
                p,
                id_manager=id_manager,
                latency_range=latency_range,
                discard_node_with_type_0=True,
                discard_node_with_unknown_latency_range=True,
                discard_graph_with_error_node_count=True,
            )

            sample_count += len(trace_graphs)
            drop_count += sum(g is None for g in trace_graphs)
            trace_graphs = [g for g in trace_graphs if g is not None]

            # evaluate the NLLs
            G = TraceGraphBatch(
                id_manager=id_manager,
                latency_range=latency_range,
                trace_graphs=trace_graphs,
            )
            chain = vae.q(G=G, n_z=eval_n_z). \
                chain(vae.p, n_z=eval_n_z, latent_axis=0, use_biased=use_biased)
            eval_nlls = T.to_numpy(chain.vi.evaluation.is_loglikelihood(reduction='none'))

            # purge too-low NLL graphs
            for g, nll in zip(trace_graphs, eval_nlls):
                if nll >= nll_threshold:
                    drop_count += 1
                else:
                    add_sample(g)

    # save the results
    drop_rate = float(drop_count / sample_count)
    result_dict.update({
        'drop_rate': drop_rate,
    })
    pprint(result_dict)

    if output_file is not None:
        _, ext = os.path.splitext(output_file)
        if ext == '.json':
            result_cont = json.dumps(result_dict)
        else:
            result_cont = yaml.safe_dump(result_dict)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_cont)

    # plot latency hist
    if latency_hist_out is not None:
        if latency_hist_out == ':show:':
            latency_hist_out = None
        plot_latency_hist(
            latency_map,
            id_manager=id_manager,
            output_file=latency_hist_out,
        )

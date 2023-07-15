from pprint import pprint

import mltk
import tensorkit as tk
from tensorkit import tensor as T

from tracegnn.data import *
from tracegnn.models.vgae.dataset import TraceGraphDataStream
from tracegnn.models.vgae.evaluation import *
from tracegnn.models.vgae.graph_utils import *
from tracegnn.models.vgae.test_utils import *
from tracegnn.models.vgae.types import TraceGraphBatch
from tracegnn.utils import *
from tracegnn.visualize import *

from tqdm import tqdm


@click.group()
def main():
    pass


@main.command(context_settings=dict(
    ignore_unknown_options=True,
    help_option_names=[],
))
@click.option('-D', '--dataset', required=True)
@click.option('-M', '--model-path', required=True)
@click.option('-o', '--nll-out', required=False, default=None)
@click.option('--proba-out', default=None, required=False)
@click.option('--auc-out', default=None, required=False)
@click.option('--latency-out', default=None, required=False)
@click.option('--gui', is_flag=True, default=False, required=False)
@click.option('--device', required=False, default=None)
@click.option('--n_z', type=int, required=False, default=10)
@click.option('--batch-size', type=int, default=64)
@click.option('--clip-nll', type=float, default=100_000)
@click.option('--no-biased', is_flag=True, default=False, required=False)
@click.option('--no-latency', is_flag=True, default=False, required=False)
@click.option('--data3', is_flag=True, default=False, required=False)
@click.option('--data4', is_flag=True, default=False, required=False)
@click.option('--drop2', is_flag=True, default=False, required=False)
@click.option('--ltest', is_flag=True, default=False, required=False)
@click.option('--limit-std', is_flag=True, default=False, required=False)
@click.argument('extra_args', nargs=-1, type=click.UNPROCESSED)
def evaluate_nll(dataset, model_path, nll_out, proba_out, auc_out, latency_out,
                 gui, device, n_z, batch_size, clip_nll, 
                 no_biased, no_latency, data3, data4, drop2, ltest, limit_std,
                 extra_args):
    # check parameters
    if gui:
        proba_out = ':show:'
        auc_out = ':show:'
        latency_out = ':show:'

    with T.use_device(device or T.first_gpu_device()):
        # load the dataset
        if ltest:
            data_names = ['test', 'ltest-drop', 'ltest-latency']
            save_filename = 'baselinel.csv'
        else:
            if drop2:
                data_names = ['test', 'test-drop-anomaly2', 'test-latency-anomaly2']
                save_filename = 'baseline2.csv'
            elif data3:
                data_names = ['test', 'test-drop-anomaly3', 'test-latency-anomaly3']
                save_filename = 'baseline3.csv'
            elif data4:
                data_names = ['test', 'test-drop-anomaly4', 'test-latency-anomaly4']
                save_filename = 'baseline4.csv'
            else:
                data_names = ['test', 'test-drop-anomaly', 'test-latency-anomaly2']
                save_filename = 'baseline.csv'

        # load the dataset
        val_db, _ = open_trace_graph_db(
            f'/srv/data/tracegnn/{dataset}/processed',
            names=['val']
        )
        db, id_manager = open_trace_graph_db(
            f'/srv/data/tracegnn/{dataset}/processed',
            names=data_names
        )
        latency_range = TraceGraphLatencyRangeFile(
            id_manager.root_dir,
            require_exists=True,
        )

        val_stream = TraceGraphDataStream(
            val_db, id_manager=id_manager, batch_size=batch_size,
            shuffle=False, skip_incomplete=False,
        )
        test_stream = TraceGraphDataStream(
            db, id_manager=id_manager, batch_size=batch_size,
            shuffle=False, skip_incomplete=False,
        )

        # load the model
        vae = load_model(
            model_path=model_path,
            id_manager=id_manager,
            strict=False,
            extra_args=extra_args,
        )
        mltk.print_config(vae.config, title='Model Config')
        vae = vae.to(T.current_device())

        # Validate model to get result_dict
        def validate():
            std_result_dict = {}

            def val_step(trace_graphs):
                with tk.layers.scoped_eval_mode(vae), T.no_grad():
                    G = TraceGraphBatch(
                        id_manager=id_manager,
                        latency_range=latency_range,
                        trace_graphs=trace_graphs,
                    )
                    chain = vae.q(G).chain(
                        vae.p,
                        G=G,
                        std_result_dict=std_result_dict
                    )
                    loss = chain.vi.training.sgvb()
                    return {'loss': T.to_numpy(T.reduce_mean(loss))}

            for [trace_graphs] in tqdm(val_stream, desc='Validation', total=val_stream.batch_count):
                val_step(trace_graphs)

            return std_result_dict

        if limit_std:
            std_result_dict = validate()
            std_limit_tensor = T.ones(
                [id_manager.num_operations], dtype=T.float32) * 1e5

            for k, v in std_result_dict.items():
                std_limit_tensor[k] = np.percentile(v, 99)
        else:
            std_limit_tensor = None

        # do evaluation
        with tk.layers.scoped_eval_mode(vae), T.no_grad():
            result_dict = do_evaluate_nll(
                test_stream=test_stream,
                vae=vae,
                id_manager=id_manager,
                latency_range=latency_range,
                n_z=n_z,
                use_biased=not no_biased,
                no_latency=no_latency,
                clip_nll=clip_nll,
                use_embeddings=False,
                nll_output_file=ensure_parent_exists(nll_out),
                proba_cdf_file=ensure_parent_exists(proba_out),
                auc_curve_file=ensure_parent_exists(auc_out),
                latency_hist_file=ensure_parent_exists(latency_out),
                dataset_name=dataset,
                save_filename=save_filename,
                std_max_limit=std_limit_tensor
            )
        pprint(result_dict)


@main.command(context_settings=dict(
    ignore_unknown_options=True,
    help_option_names=[],
))
@click.option('-D', '--data-dir', required=True)
@click.option('-M', '--model-path', required=True)
@click.option('-o', '--output-file', required=False, default=None)
@click.option('--gui', is_flag=True, default=False, required=False)
@click.option('--latency-hist-out', required=False, default=None)
@click.option('--device', required=False, default=None)
@click.option('--n_samples', type=int, required=False, default=1_0000)
@click.option('--batch-size', type=int, default=64)
@click.option('--eval-n_z', type=int, required=False, default=10)
@click.option('--no-biased', is_flag=True, default=False, required=False)
@click.option('--nll-threshold', type=float, required=True)
@click.argument('extra_args', nargs=-1, type=click.UNPROCESSED)
def evaluate_prior(data_dir, model_path, output_file, gui, latency_hist_out,
                   device, n_samples, batch_size, eval_n_z, no_biased, nll_threshold,
                   extra_args):
    # check parameters
    if gui:
        latency_hist_out = ':show:'

    with T.use_device(device or T.first_gpu_device()):
        # load the dataset
        db, id_manager = open_trace_graph_db(
            data_dir,
            names=['test']
        )
        latency_range = TraceGraphLatencyRangeFile(
            id_manager.root_dir,
            require_exists=True,
        )

        # load the model
        vae = load_model(
            model_path=model_path,
            id_manager=id_manager,
            strict=False,
            extra_args=extra_args,
        )
        mltk.print_config(vae.config, title='Model Config')
        vae = vae.to(T.current_device())

        # do evaluation
        with tk.layers.scoped_eval_mode(vae), T.no_grad():
            do_evaluate_prior(
                vae=vae,
                id_manager=id_manager,
                latency_range=latency_range,
                n_samples=n_samples,
                batch_size=batch_size,
                eval_n_z=eval_n_z,
                nll_threshold=nll_threshold,
                use_biased=not no_biased,
                output_file=ensure_parent_exists(output_file),
                latency_hist_out=ensure_parent_exists(latency_hist_out),
            )


if __name__ == '__main__':
    main()

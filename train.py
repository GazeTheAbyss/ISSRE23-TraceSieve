import json
import math
import random
import shutil
from enum import Enum
from typing import *

import mltk
import tensorkit as tk
import torch
from tensorkit import tensor as T
from tensorkit.examples import utils
from tensorkit.train import Checkpoint
from tensorkit.backend.pytorch_.optim import BackendOptimizer

from tracegnn.data import *
from tracegnn.models.vgae.evaluation import *
from tracegnn.models.vgae.graph_utils import *
from tracegnn.models.vgae.tensor_utils import get_moments
from tracegnn.models.vgae.types import *
from tracegnn.models.vgae.model import *
from tracegnn.models.vgae.dataset import *
from tracegnn.utils import *
from tracegnn.visualize import *

class NANLossError(Exception):

    def __init__(self, epoch):
        super().__init__(epoch)

    @property
    def epoch(self) -> Optional[int]:
        return self.args[0]

    def __str__(self):
        return f'NaN loss encountered at epoch {self.epoch}'


class OptimizerType(str, Enum):
    ADAM = 'adam'
    RMSPROP = 'rmsprop'


class ExpConfig(mltk.Config):
    model: TraceVAEConfig = TraceVAEConfig()
    device: Optional[str] = 'cpu'
    seed: Optional[int] = None

    class train(mltk.Config):
        max_epoch: int = 60
        ckpt_epoch_freq: Optional[int] = 5
        test_epoch_freq: Optional[int] = 5
        latency_hist_epoch_freq: Optional[int] = 10

        use_early_stopping: bool = False
        val_epoch_freq: Optional[int] = 2

        kl_beta: float = 0.5
        warm_up_epochs: Optional[int] = None  # number of epochs used to warm-up the prior (KLD)

        l2_reg: float = 0.0001
        z_unit_ball_reg: Optional[float] = None
        z2_unit_ball_reg: Optional[float] = None

        init_batch_size: int = 64
        batch_size: int = 64
        val_batch_size: int = 64

        optimizer: OptimizerType = OptimizerType.RMSPROP
        initial_lr: float = 0.001
        lr_anneal_ratio: float = 0.1
        lr_anneal_epochs: int = 20
        clip_norm: Optional[float] = None
        global_clip_norm: Optional[float] = 10  # important for numerical stability

        latency_logstd_anneal: bool = False
        initial_latency_logstd: float = -1
        latency_logstd_anneal_epochs: Optional[int] = 5

        test_n_z: int = 10
        num_plot_samples: int = 20

    class test(mltk.Config):
        batch_size: int = 64
        eval_n_z: int = 10
        use_biased: bool = True
        clip_nll: Optional[float] = 100_000
        data3: bool = False
        data4: bool = True
        drop2: bool = False
        ltest: bool = False

    class report(mltk.Config):
        html_ext: str = '.html.gz'

    class dataset(mltk.Config):
        root_dir: str = os.path.abspath('./data/processed')


class RMSprop(BackendOptimizer):

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0,
                 momentum=0, centered=False):
        params = list(params)
        super().__init__(
            params=params,
            lr=lr,
            torch_optimizer_factory=lambda: torch.optim.RMSprop(
                params=params,
                lr=lr,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                momentum=momentum,
                centered=centered,
            )
        )


def main(exp: mltk.Experiment[ExpConfig]):
    # config
    config = exp.config

    # set random seed to encourage reproducibility (does it really work?)
    if config.seed is not None:
        T.random.set_deterministic(True)
        T.random.seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

    # Load data
    id_manager = TraceGraphIDManager(config.dataset.root_dir)
    latency_range = TraceGraphLatencyRangeFile(config.dataset.root_dir)

    train_db = TraceGraphDB(BytesSqliteDB(os.path.join(config.dataset.root_dir, 'train')))
    val_db = TraceGraphDB(BytesSqliteDB(os.path.join(config.dataset.root_dir, 'val')))

    # load the dataset
    if config.test.ltest:
        data_names = ['test', 'ltest-drop', 'ltest-latency']
        save_filename = 'baselinel.csv'
    else:
        if config.test.drop2:
            data_names = ['test', 'test-drop-anomaly2', 'test-latency-anomaly2']
            save_filename = 'baseline2.csv'
        elif config.test.data3:
            data_names = ['test', 'test-drop-anomaly3', 'test-latency-anomaly3']
            save_filename = 'baseline3.csv'
        elif config.test.data4:
            data_names = ['test', 'test-drop-anomaly4', 'test-latency-anomaly4']
            save_filename = 'baseline4.csv'
        else:
            data_names = ['test', 'test-drop-anomaly', 'test-latency-anomaly2']
            save_filename = 'baseline.csv'

    test_db = TraceGraphDB(
        BytesMultiDB(*[
            BytesSqliteDB(os.path.join(config.dataset.root_dir, file_path))
                for file_path in data_names]
        )
    )

    train_stream = TraceGraphDataStream(
        train_db, id_manager=id_manager, batch_size=config.train.batch_size,
        shuffle=True, skip_incomplete=False
    )
    val_stream = TraceGraphDataStream(
        val_db, id_manager=id_manager, batch_size=config.train.val_batch_size,
        shuffle=False, skip_incomplete=False,
    )
    test_stream = TraceGraphDataStream(
        test_db, id_manager=id_manager, batch_size=config.test.batch_size,
        shuffle=False, skip_incomplete=False,
    )

    utils.print_experiment_summary(exp, train_data=train_stream, val_data=val_stream, test_data=test_stream)

    # build the network
    vae: TraceVAE = TraceVAE(
        config.model,
        id_manager.num_operations,
    )
    vae = vae.to(T.current_device())
    params, param_names = utils.get_params_and_names(vae)
    utils.print_parameters_summary(params, param_names)
    print('')
    mltk.print_with_time('Network constructed.')

    # define the train and evaluate functions
    def get_latency_logstd_min():
        if config.train.latency_logstd_anneal:
            return config.train.initial_latency_logstd - (
                (loop.epoch - 1) // config.train.latency_logstd_anneal_epochs
            )

    def train_step(trace_graphs):
        G = TraceGraphBatch(
            id_manager=id_manager,
            latency_range=latency_range,
            trace_graphs=trace_graphs,
        )
        chain = vae.q(G).chain(
            vae.p,
            G=G,
            latency_logstd_min=get_latency_logstd_min(),
        )

        # collect the log likelihoods
        p_obs = []
        p_latent = []
        q_latent = []
        for name in chain.p:
            if name in chain.q:
                q_latent.append(chain.q[name].log_prob())
                p_latent.append(chain.p[name].log_prob())
            else:
                p_obs.append(chain.p[name].log_prob())

        # get E[log p(x|z)] and KLD[q(z|x)||p(z)]
        recons = T.reduce_mean(T.add_n(p_obs))
        kl = T.reduce_mean(T.add_n(q_latent) - T.add_n(p_latent))

        # KL beta
        beta = config.train.kl_beta
        if config.train.warm_up_epochs and loop.epoch < config.train.warm_up_epochs:
            beta = beta * (loop.epoch / config.train.warm_up_epochs)
        loss = beta * kl - recons

        # l2 regularization
        if config.train.l2_reg:
            loss = loss + config.train.l2_reg * T.nn.l2_regularization(params)

        # unit ball regularization
        def add_unit_ball_reg(l, t, reg):
            if reg is not None:
                ball_mean, ball_var = get_moments(t, axis=[-1])
                l = l + reg * (
                    T.reduce_mean(ball_mean ** 2) +
                    T.reduce_mean((ball_var - 1) ** 2)
                )
            return l

        loss = add_unit_ball_reg(loss, chain.q['z'].tensor, config.train.z_unit_ball_reg)
        if 'z2' in chain.q:
            loss = add_unit_ball_reg(loss, chain.q['z2'].tensor, config.train.z2_unit_ball_reg)

        # check and return the metrics
        if math.isnan(T.to_numpy(loss)):
            raise NANLossError(loop.epoch)
        return {'loss': loss, 'recons': recons, 'kl': kl}

    def validate():
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
                    latency_logstd_min=get_latency_logstd_min(),
                )
                loss = chain.vi.training.sgvb()
                return {'loss': T.to_numpy(T.reduce_mean(loss))}

        val_loop = loop.validation()
        result_dict = val_loop.run(val_step, val_stream)
        summary_cb.update_metrics({
            f'val_{k}': v
            for k, v in result_dict.items()
        })

    def evaluate(n_z, eval_loop, eval_stream, epoch, use_embeddings=False,
                 plot_latency_hist=False, save_filename=None):
        # latency_hist_file
        latency_hist_file = None
        if plot_latency_hist:
            latency_hist_file = exp.make_parent(f'./plotting/latency-sample/{epoch}.jpg')

        # do evaluation
        with tk.layers.scoped_eval_mode(vae), T.no_grad():
            result_dict = do_evaluate_nll(
                test_stream=eval_stream,
                vae=vae,
                id_manager=id_manager,
                latency_range=latency_range,
                n_z=n_z,
                use_biased=config.test.use_biased,
                test_loop=eval_loop,
                summary_writer=summary_cb,
                clip_nll=config.test.clip_nll,
                use_embeddings=use_embeddings,
                nll_output_file=exp.make_parent(f'./result/test-anomaly/{epoch}.npz'),
                proba_cdf_file=exp.make_parent(f'./plotting/test-nll/{epoch}.jpg'),
                auc_curve_file=exp.make_parent(f'./plotting/test-auc/{epoch}.jpg'),
                latency_hist_file=latency_hist_file,
                dataset_name=config.dataset.root_dir.split('/')[-2],
                save_filename=save_filename
            )

        with open(exp.make_parent(f'./result/test-anomaly/{epoch}.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(result_dict))
        eval_loop.add_metrics(**result_dict)

    def plot_recons(epoch=None):
        epoch = epoch or loop.epoch
        graph_ids = []
        trace_graphs = []
        for i, g in train_db.sample_n(config.train.num_plot_samples, with_id=True):
            graph_ids.append(i)
            trace_graphs.append(g)

        with tk.layers.scoped_eval_mode(vae), T.no_grad():
            G = TraceGraphBatch(
                id_manager=id_manager,
                latency_range=latency_range,
                trace_graphs=trace_graphs,
            )
            chain = vae.q(G).chain(
                vae.p,
                G=G,
                latency_logstd_min=get_latency_logstd_min(),
            )
            graph_list = []
            recons_graphs = flat_to_nx_graphs(chain.p, id_manager, latency_range)
            for i, a, b in zip(graph_ids, trace_graphs, recons_graphs):
                graph_list.append((f'#{i}', a))
                graph_list.append((f'#{i}-recons', b))

            render_trace_graph_html(
                graph_list,
                id_manager=id_manager,
                output_file=exp.make_parent(f'plotting/recons/{epoch}{config.report.html_ext}')
            )

    def plot_samples(epoch=None):
        epoch = epoch or loop.epoch
        with tk.layers.scoped_eval_mode(vae), T.no_grad():
            graph_list = flat_to_nx_graphs(
                vae.p(n_z=config.train.num_plot_samples),
                id_manager,
                latency_range
            )
            render_trace_graph_html(
                graph_list,
                id_manager=id_manager,
                output_file=exp.make_parent(f'plotting/samples/{epoch}{config.report.html_ext}')
            )

    def save_model(epoch=None):
        epoch = epoch or loop.epoch
        torch.save(vae.state_dict(), exp.make_parent(f'models/{epoch}.pt'))

    # the train loop
    loop = mltk.TrainLoop(max_epoch=config.train.max_epoch)
    # validate()
    # loop.add_callback(mltk.callbacks.StopOnNaN())

    # the summary writer
    summary_cb = SummaryCallback(summary_dir=exp.abspath('./summary'))
    loop.add_callback(summary_cb)

    # plot_samples()
    # evaluate(10, mltk.TestLoop(), test_stream, 0)

    # the optimizer and learning rate scheduler
    train_params = list(tk.layers.iter_parameters(vae))
    if config.train.optimizer == OptimizerType.ADAM:
        optimizer = tk.optim.Adam(train_params)
    elif config.train.optimizer == OptimizerType.RMSPROP:
        optimizer = RMSprop(train_params)

    lr_scheduler = tk.optim.lr_scheduler.AnnealingLR(
        optimizer=optimizer,
        initial_lr=config.train.initial_lr,
        ratio=config.train.lr_anneal_ratio,
        epochs=config.train.lr_anneal_epochs,
    )
    lr_scheduler.bind(loop)

    # checkpoint
    ckpt = Checkpoint(vae=vae)
    loop.add_callback(mltk.callbacks.AutoCheckpoint(
        ckpt,
        root_dir=exp.make_dirs('./checkpoint'),
        epoch_freq=config.train.ckpt_epoch_freq,
        max_checkpoints_to_keep=10,
    ))

    # install the plot and sample functions during training
    # loop.run_after_every(plot_recons, epochs=1)
    # loop.run_after_every(plot_samples, epochs=1)
    loop.run_after_every(save_model, epochs=1)

    # install the validation function and early-stopping
    if config.train.val_epoch_freq:
        loop.run_after_every(
            validate,
            epochs=config.train.val_epoch_freq,
        )
        if config.train.use_early_stopping:
            loop.add_callback(mltk.callbacks.EarlyStopping(
                checkpoint=ckpt,
                root_dir=exp.abspath('./early-stopping'),
                metric_name='val_loss',
            ))

    # install the evaluation function during training
    if config.train.test_epoch_freq:
        loop.run_after_every(
            lambda: evaluate(
                n_z=config.train.test_n_z,
                eval_loop=loop.test(),
                eval_stream=test_stream,
                epoch=loop.epoch,
                plot_latency_hist=(
                    config.train.latency_hist_epoch_freq and
                    loop.epoch % config.train.latency_hist_epoch_freq == 0
                )
            ),
            epochs=config.train.test_epoch_freq,
        )

    # initialize the model
    G = TraceGraphBatch(
        id_manager=id_manager,
        latency_range=latency_range,
        trace_graphs=train_db.sample_n(config.train.init_batch_size),
    )
    chain = vae.q(G).chain(
        vae.p,
        G=G,
        latency_logstd_min=get_latency_logstd_min(),
    )
    loss = chain.vi.training.sgvb(reduction='mean')
    mltk.print_with_time(f'Network initialized: loss = {T.to_numpy(loss)}')

    # train the model
    try:
        tk.layers.set_train_mode(vae, True)
        utils.fit_model(
            loop=loop,
            optimizer=optimizer,
            fn=train_step,
            stream=train_stream,
            clip_norm=config.train.clip_norm,
            global_clip_norm=config.train.global_clip_norm,
        )
    except KeyboardInterrupt:
        print('Train interrupted, press Ctrl+C again to skip the final test ...', file=sys.stderr)

    # save the final model
    save_model('final')

    # do final evaluation
    # plot_recons('final')
    # plot_samples('final')
    evaluate(
        n_z=config.test.eval_n_z,
        eval_loop=mltk.TestLoop(),
        eval_stream=test_stream,
        epoch='final',
        use_embeddings=True,
        plot_latency_hist=True,
        save_filename=save_filename
    )


if __name__ == '__main__':
    with mltk.Experiment(ExpConfig) as exp:
        config = exp.config
        device = config.device or T.first_gpu_device()
        with T.use_device(device):
            retrial = 0
            while True:
                try:
                    main(exp)
                except NANLossError as ex:
                    if ex.epoch > 1 or retrial >= 10:
                        raise
                    retrial += 1
                    print(
                        f'\n'
                        f'Restart the experiment for the {retrial}-th time '
                        f'due to NaN loss at epoch {ex.epoch}.\n',
                        file=sys.stderr
                    )
                    for name in ['checkpoint', 'early-stopping', 'models',
                                 'plotting', 'summary']:
                        path = exp.abspath(name)
                        if os.path.isdir(name):
                            shutil.rmtree(path)
                else:
                    break

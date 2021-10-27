from common import *
from metrics import Metric
from tracking import Stage
from tensorboard_experiment import TensorboardExperiment

from typing import Any
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class AvgStats():
    def __init__(self, metrics, in_train): self.metrics,self.in_train = listify(metrics),in_train

    def reset(self):
        self.tot_loss,self.count = 0.,0
        self.tot_mets = [0.] * len(self.metrics)

    @property
    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets
    @property
    def avg_stats(self): return [o/self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn


class Callback():
    _order=0
    def set_runner(self, run): self.run=run
    def __getattr__(self, k): return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False


class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs=0.
        self.run.n_iter=0

    def after_batch(self):
        if not self.in_train: return
        self.run.n_epochs += 1./self.iters
        self.run.n_iter   += 1

    def begin_epoch(self):
        self.run.n_epochs=self.epoch
        self.model.train()
        self.run.in_train=True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train=False


class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)

    def after_epoch(self):
        if self.run.epoch % 10==0:
            print(f'epoch: {self.run.epoch + 10}')
            print(self.train_stats)
            print(self.valid_stats)


class Recorder(Callback):
    def begin_fit(self):
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []

    def after_batch(self):
        if not self.in_train: return
        for pg,lr in zip(self.opt.param_groups,self.lrs): lr.append(pg['lr'])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr  (self, pgid=-1): plt.plot(self.lrs[pgid])
    def plot_loss(self, skip_last=0): plt.plot(self.losses[:len(self.losses)-skip_last])

    def plot(self, skip_last=0, pgid=-1):
        losses = [o.item() for o in self.losses]
        lrs    = self.lrs[pgid]
        n = len(losses)-skip_last
        plt.xscale('log')
        plt.plot(lrs[:n], losses[:n])


class ParamScheduler(Callback):
    _order=1
    def __init__(self, pname, sched_funcs): self.pname,self.sched_funcs = pname,sched_funcs

    def begin_fit(self):
        if not isinstance(self.sched_funcs, (list,tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        assert len(self.opt.param_groups)==len(self.sched_funcs)
        for pg,f in zip(self.opt.param_groups,self.sched_funcs):
            pg[self.pname] = f(self.n_epochs/self.epochs)

    def begin_batch(self):
        if self.in_train: self.set_param()


class LR_Find(Callback):
    _order=1
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if not self.in_train: return
        pos = self.n_iter/self.max_iter
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in self.opt.param_groups: pg['lr'] = lr

    def after_step(self):
        if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:
            raise CancelTrainException()
        if self.loss < self.best_loss: self.best_loss = self.loss


class Tracker(Callback):
    def __init__(self, run_path):
        self.tracker = TensorboardExperiment(log_path=run_path)
        self.run_count = 0
        self.accuracy_metric = Metric()
        self.y_true_batches: list[list[Any]] = []
        self.y_pred_batches: list[list[Any]] = []

    @property
    def avg_accuracy(self):
        return self.accuracy_metric.average

    def begin_batch(self):
        self.run_count += 1
        self.batch_size: int = self.run.xb.shape[0]

    def after_loss(self):
        # Compute Batch Validation Metrics
        y_np = self.run.yb.detach().numpy()
        y_prediction_np = np.argmax(self.pred.detach().numpy(), axis=1)
        self.batch_accuracy = accuracy_score(y_np, y_prediction_np)
        self.accuracy_metric.update(self.batch_accuracy, self.batch_size)
        self.y_true_batches += [y_np]
        self.y_pred_batches += [y_prediction_np]

    def after_batch(self):
        batch_accuracy = self.batch_accuracy
        run_count = self.run_count
        self.tracker.add_batch_metric("accuracy", batch_accuracy, run_count)

    def after_epoch(self):
        if self.in_train:
            self.tracker.add_epoch_metric("accuracy", self.avg_accuracy, self.run_count)
        else:
            self.tracker.add_epoch_metric("accuracy", self.avg_accuracy, self.run_count)
            self.tracker.add_epoch_confusion_matrix(
                    self.y_true_batches, self.y_pred_batches, self.run_count
            )

    def begin_epoch(self):
        self.tracker.set_stage(Stage.TRAIN)
        self.run.n_epochs=self.epoch
        self.model.train()
        self.run.in_train=True

    def begin_validate(self):
        self.tracker.set_stage(Stage.VAL)
        self.model.eval()
        self.run.in_train=False

def accuracy(out, yb):
    return (torch.argmax(out, dim=1)==yb).float().mean()

def adjusted_accu(out, yb):
    return (F.log_softmax(out, dim=1).argmax(dim=1)==yb).float().mean()
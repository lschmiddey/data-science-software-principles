from callbacks import *

from typing import Any
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model,self.opt,self.loss_func,self.data = model,opt,loss_func,data


class Runner():
    def __init__(self, cbs=None, cb_funcs=None):
        self.in_train = False
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop,self.cbs = False,[TrainEvalCallback()]+cbs

    @property
    def opt(self):       return self.learn.opt
    @property
    def model(self):     return self.learn.model
    @property
    def loss_func(self): return self.learn.loss_func
    @property
    def data(self):      return self.learn.data

    def one_batch(self, xb, emb, yb):
        try:
            self.xb,self.emb,self.yb = xb,emb,yb
            self('begin_batch')
            self.pred = self.model(self.xb,self.emb)
            self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb)
            self('after_loss')
            if not self.in_train: return
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        except CancelBatchException: self('after_cancel_batch')
        finally: self('after_batch')

    def all_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb,emb,yb in dl: self.one_batch(xb, emb, yb)
        except CancelEpochException: self('after_cancel_epoch')

    def fit(self, epochs, learn):
        self.epochs,self.learn,self.loss = epochs,learn,torch.tensor(0.)

        try:
            for cb in self.cbs: cb.set_runner(self)
            self('begin_fit')
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'): self.all_batches(self.data.train_dl)

                with torch.no_grad():
                    if not self('begin_validate'): self.all_batches(self.data.valid_dl)
                self('after_epoch')

        except CancelTrainException: self('after_cancel_train')
        finally:
            self('after_fit')
            self.learn = None

    def predict(self, pred_data, learn):
        self.pred_data, self.learn = pred_data, learn
        self.preds_array = np.array([]).astype(int)
        self.model.eval()
        with torch.no_grad():
            for batch in self.pred_data:
                xb, emb, *_ = batch
                out = self.model(xb,emb)
                preds = F.log_softmax(out, dim=1).argmax(dim=1).numpy()
                self.preds_array = np.concatenate((self.preds_array, preds), axis=None)
        return self.preds_array

    def check_results(self, pred_data, learn):
        self.pred_data, self.learn = pred_data, learn
        self.preds_array = np.array([]).astype(int)
        self.y_test = np.array([]).astype(int)
        self.target_names = ['no Churn', 'Churn']
        self.model.eval()
        with torch.no_grad():
            for batch in self.pred_data:
                xb, emb, yb, _ = batch
                out = self.model(xb,emb)
                preds = F.log_softmax(out, dim=1).argmax(dim=1).numpy()
                self.preds_array = np.concatenate((self.preds_array, preds), axis=None)
                self.y_test = np.concatenate((self.y_test, yb), axis=None)

        self.conf_matrix = confusion_matrix(self.y_test, self.preds_array)
        self.class_report = classification_report(self.y_test, self.preds_array, target_names=self.target_names)
        return self.conf_matrix, self.class_report

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) or res
        return res

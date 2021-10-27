#%%
from transform_data import *
from common import *
from dataloaders import *
from config import *
from model_part import *
from callbacks import *
from runner import *

from functools import partial
import pandas as pd
from torch import optim
#%%

df_train = pd.read_csv('data/ItalyPowerDemand_TRAIN.txt', header=None,delim_whitespace=True)
df_test = pd.read_csv('data/ItalyPowerDemand_TEST.txt', header=None, delim_whitespace=True)
df_train.head()
# %%
x_train, x_test, y_train, y_test, emb_vars_train, emb_vars_test = transform_data(df_train, df_test)

#%%
emb_vars_train, emb_vars_test, dict_embs, dict_inv_embs = cat_transform(emb_vars_train, emb_vars_test)

# %%
device = DEVICE
datasets = create_datasets(x_train, emb_vars_train, y_train,
             x_test, emb_vars_test, y_test,
             valid_pct=VAL_SIZE, seed=1234)
data = DataBunch(*create_loaders(datasets, bs=1024))
# %%
# define model
raw_feat = x_train.shape[1]
emb_dims = [(len(dict_embs[0]), EMB_DIMS), (len(dict_embs[1]), EMB_DIMS)]

num_classes = 2

# create model
model = Classifier(nn.Sequential(
    *get_cnn_layers(raw_feat, OUTPUT_SHAPES, KERNELS, STRIDES)
    ), emb_dims, num_classes).to(device)
opt = optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

# %%
learn = Learner(model, opt, loss_func, data)
run = Runner(cb_funcs=[LR_Find, Recorder])

# %%
%matplotlib inline

run.fit(100, learn)
run.recorder.plot(skip_last=5)

# %%
model = Classifier(nn.Sequential(
    *get_cnn_layers(raw_feat, OUTPUT_SHAPES, KERNELS, STRIDES)
    ), emb_dims, num_classes).to(device)
opt = optim.Adam(model.parameters(), lr=2e-3)

cbfs = [Recorder, partial(AvgStatsCallback,adjusted_accu), partial(Tracker, RUN_PATH)]
learn = Learner(model, opt, loss_func, data)
run = Runner(cb_funcs=cbfs)
run.fit(40, learn)
# %%
run.recorder.plot_loss()
# %%

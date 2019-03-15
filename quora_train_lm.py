from fastai.text import *
from fastai.callback import *
from fastai.imports import *
from fastai.callbacks.tracker import SaveModelCallback

path = Path('./quora')
bs = 128


data_lm = load_data(path, 'data_lm.pkl', bs=bs)
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

learn.load('fit_head')

learn.unfreeze()

learn.callback_fns.append(partial(
    SaveModelCallback, every='improvement', name='best_fine_tuned' ))

learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))

learn.load('best_fine_tuned')
learn.save_encoder('fine_tuned_enc')
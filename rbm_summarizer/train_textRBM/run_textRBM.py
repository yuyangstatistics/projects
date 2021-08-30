from __future__ import unicode_literals, print_function, division
# load modules in other files
import sys
import os
import time

# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

# # autoreload self-defined modules
# %load_ext autoreload
# %autoreload 2

from data_util import utils
from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from training_ptr_gen.train_util import get_rbm_input_from_batch

from textRBM import TextRBM
from training_ptr_gen.train import Train


def save(save_dir, iter, model):
    ckpt_dict = {'model_state': model.state_dict(), 'iter': iter}
    ckpt_path = os.path.join(save_dir, "iter_%d.pth.tar" % iter)
    torch.save(ckpt_dict, ckpt_path)

if __name__ == '__main__':

    vocab = Vocab(config.vocab_path, config.vocab_size)
    batcher = Batcher(config.train_data_path, vocab, mode='train',
                            batch_size=128, single_pass=False)
    time.sleep(15)

    import time
    start_time = time.time()
    save_dir = "/home/yang6367/gitrepos/pointer_summarizer/train_textRBM/save/rbm%d" % int(start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    use_cuda = True
    device = utils.get_devices([5])
    rbm = TextRBM(k=1, device=device).to(device)
    train_op = optim.Adagrad(rbm.parameters(), lr=0.15, initial_accumulator_value=config.adagrad_init_acc)

    iter = 0
    niters = 5000
    loss_ = []
    while iter < niters:
        batch = batcher.next_batch()
        docs_word_count = get_rbm_input_from_batch(batch, vocab, use_cuda, device)

        v,v1 = rbm(docs_word_count.float())
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        loss_.append(loss.data)
        train_op.zero_grad()
        loss.backward()
        clip_grad_norm_(rbm.parameters(), config.max_grad_norm)
        train_op.step()

        iter += 1

        # update k
        if iter % 1000 == 0:
            rbm.k = int(iter / 1000) + 1
        if iter % 50 == 0:
            print("Training loss for %d iter: %.5f" % (iter, np.mean(loss_)))
            loss_ = []
        if iter % 200 == 0:
            save(save_dir, iter, rbm)

    print("%f seconds" % (time.time() - start_time))


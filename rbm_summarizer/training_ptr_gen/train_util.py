from torch.autograd import Variable
import numpy as np
import torch
from data_util import config
from data_util import utils
from train_textRBM.textRBM import TextRBM
import torch

def get_input_from_batch(batch, use_cuda, device='cpu'):
  batch_size = len(batch.enc_lens)

  enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
  enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
  enc_lens = Variable(torch.from_numpy(batch.enc_lens)).int()
  extra_zeros = None
  enc_batch_extend_vocab = None

  if config.pointer_gen:
    enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
    # max_art_oovs is the max over all the article oov list in the batch
    if batch.max_art_oovs > 0:
      extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

  c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

  coverage = None
  if config.is_coverage:
    coverage = Variable(torch.zeros(enc_batch.size()))

  if use_cuda:
    enc_batch = enc_batch.to(device)
    enc_padding_mask = enc_padding_mask.to(device)
    enc_lens = enc_lens.to(device)

    if enc_batch_extend_vocab is not None:
      enc_batch_extend_vocab = enc_batch_extend_vocab.to(device)
    if extra_zeros is not None:
      extra_zeros = extra_zeros.to(device)
    c_t_1 = c_t_1.to(device)

    if coverage is not None:
      coverage = coverage.to(device)

  return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage

def get_output_from_batch(batch, use_cuda, device='cpu'):
  dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
  dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
  dec_lens = batch.dec_lens
  max_dec_len = np.max(dec_lens)
  dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()

  target_batch = Variable(torch.from_numpy(batch.target_batch)).long()

  if use_cuda:
    dec_batch = dec_batch.to(device)
    dec_padding_mask = dec_padding_mask.to(device)
    dec_lens_var = dec_lens_var.to(device)
    target_batch = target_batch.to(device)


  return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch


def get_rbm_input_from_batch(batch, vocab, use_cuda, device):
  ## step 1: get the cleaned word list for each document
  # convert the indices to words
  docs = [[vocab.id2word(i) for i in doc_ids ]for doc_ids in batch.enc_batch.tolist()]
  # remove stopwords, meaningless words, and do lemmatization
  docs = [utils.lemmatize(utils.clean(doc)) for doc in docs]
  # filter out the words that are not in the top15k
  docs = [utils.keep_top15k(doc) for doc in docs]

  ## step 2: transform the word list into a word-count vector
  docs_word_count = [utils.get_word_count(doc) for doc in docs]
  # transform the list to a tensor
  docs_word_count = Variable(torch.Tensor(docs_word_count).int())
  # docs_lens = docs_word_count.sum(-1)
  
  if use_cuda:
    docs_word_count = docs_word_count.to(device)
    # docs_lens = docs_lens.to(device)
  
  return docs_word_count


# load the trained rbm model
ckpt_dict = torch.load(config.rbm_ckpt_path, map_location="cpu")
rbm = TextRBM(k=1)
rbm.load_state_dict(ckpt_dict['model_state'])
# device = utils.get_devices(config.gpu_ids)
# rbm.to(device)

def get_rbm_output_from_batch(batch, vocab, use_cuda, device):
  
  docs_word_count = get_rbm_input_from_batch(batch, vocab, False, 'cpu').float()
  with torch.no_grad():
    _, sample_h = rbm.v_to_h(docs_word_count, docs_word_count.sum(-1).int())

  if use_cuda:
    sample_h = sample_h.to(device)

  return sample_h  # B x latent_dim (200)

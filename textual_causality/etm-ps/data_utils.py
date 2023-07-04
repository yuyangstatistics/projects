from torch.utils.data import Dataset
import os
import json
import torch
from transformers import RobertaTokenizer
from collections import Counter
import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

class CNNDM(Dataset):
    def __init__(self, fdir, model_type, is_test=False, total_len=512, ps_path=None):
        """ data format: article, abstract, [(candidiate_i, score_i)] 
        ps_path: the learned propensity score dictionary json file.
        """
        self.fdir = fdir
        self.indices = self.get_valid_indices()
        self.num = len(self.indices)
        self.is_test = is_test
        self.total_len = total_len
        self.tok = RobertaTokenizer.from_pretrained(model_type, verbose=False)
        self.pad_token_id = self.tok.pad_token_id
        self.cls_token_id = self.tok.cls_token_id
        self.sep_token_id = self.tok.sep_token_id
        self.keywords = ["government", "crime", "economy", "game", "health"]
        self.ps_path = ps_path

    def __len__(self):
        return self.num

    def get_valid_indices(self):
        """There are some stories in the training set that have empty articles, which 
        will lead to NaNs in training process if not removed. (an example is train/589.json)
        """
        valid_indices = []
        for idx in range(len(os.listdir(self.fdir))):
            with open(os.path.join(self.fdir, f"{idx}.json"), "r") as f:
                data = json.load(f)
            if data["article"] and data["abstract"]:
                valid_indices.append(idx)
        return valid_indices

    def bert_encode(self, x, get_counters=False):
        _ids = self.tok.encode(x, add_special_tokens=False, truncation=False)
        ids = [self.cls_token_id]
        ids.extend(_ids[:self.total_len - 2])
        ids.append(self.sep_token_id)
        if get_counters:
            cnt = Counter(_ids)
            return ids, cnt, _ids
        else:
            return ids, _ids

    def __getitem__(self, idx):
        with open(os.path.join(self.fdir, f"{self.indices[idx]}.json"), "r") as f:
            data = json.load(f)
        article = data["article"]
        src_input_ids, src_counters, _src_input_ids = self.bert_encode(" ".join(article), get_counters=True)
        src_input_len = len(_src_input_ids)
        src_keywords_inclusion = [1 * (w in " ".join(article)) for w in self.keywords]
        src_keywords_inclusion_type = int("".join([str(x) for x in src_keywords_inclusion]), 2)  # convert a binary list into a number
        abstract = data["abstract"]
        tgt_input_ids, _ = self.bert_encode(" ".join(abstract), get_counters=False)
        text_id = self.indices[idx]
        result = {
            "src_input_ids": src_input_ids, 
            "tgt_input_ids": tgt_input_ids, 
            "src_counters": src_counters, 
            "src_input_len": src_input_len, 
            "src_keywords_inclusion_type": src_keywords_inclusion_type, 
            "text_id": text_id
        }
        if self.is_test:
            result["text"] = data
        if self.ps_path:
            pass
        return result

    def select(self, indices):
        self.num = len(indices)
        self.indices = indices
        return self

def collate_mp(examples, pad_token_id, vocab_size, is_test=False):
    def bert_pad(sents):
        max_len = max(len(sent) for sent in sents)
        result = []
        for sent in sents:
            if len(sent) < max_len:
                sent.extend([pad_token_id] * (max_len - len(sent)))
            result.append(sent)
        return torch.LongTensor(result)

    def get_bows(counters, vocab_size):
        return torch.IntTensor([[cnt[i] for i in range(vocab_size)] for cnt in counters])

    src_input_ids = bert_pad([x["src_input_ids"] for x in examples])
    tgt_input_ids = bert_pad([x["tgt_input_ids"] for x in examples])
    src_bows = get_bows([x["src_counters"] for x in examples], vocab_size)
    src_input_lens = torch.LongTensor([x["src_input_len"] for x in examples])
    src_keywords_inclusion_types = torch.IntTensor([x["src_keywords_inclusion_type"] for x in examples])
    text_ids = [x["text_id"] for x in examples]
    result = {
        "src_input_ids": src_input_ids, 
        "tgt_input_ids": tgt_input_ids, 
        "src_bows": src_bows,    # (batch_size, vocab_size)
        "src_input_lens": src_input_lens, 
        "src_keywords_inclusion_types": src_keywords_inclusion_types, 
        "text_ids": text_ids
    }
    if is_test:
        result['text'] = [x["text"] for x in examples]
    return result 




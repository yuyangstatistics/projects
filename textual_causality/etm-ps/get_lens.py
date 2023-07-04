"""This file is to get the document length distribution."""
from transformers import RobertaTokenizer
from functools import partial
from data_utils import collate_mp, CNNDM
from torch.utils.data import DataLoader
from collections import OrderedDict
import json

if __name__ == "__main__":

    tok = RobertaTokenizer.from_pretrained("roberta-base")
    train_dataset = CNNDM(f"/home/yang6367/summarizer/cnn-dailymail/processed/train", "roberta-base", is_test=False)
    collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, vocab_size=tok.vocab_size, is_test=False)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    doc_lens = OrderedDict()
    for batch in train_dataloader:
        doc_lens.update(zip(batch["text_ids"], batch["src_input_lens"]))
            
    with open('save/document_lengths.json', 'w') as fp:
        json.dump(doc_lens, fp)
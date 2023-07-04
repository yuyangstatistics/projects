"""This file is to get the propensity scores dictionary."""
from transformers import RobertaTokenizer
from functools import partial
from data_utils import collate_mp, CNNDM
from torch.utils.data import DataLoader
from collections import Counter, OrderedDict
from tqdm import tqdm
from json import dump

if __name__ == "__main__":

    tok = RobertaTokenizer.from_pretrained("roberta-base")
    train_dataset = CNNDM(f"/home/yang6367/summarizer/cnn-dailymail/processed/train", "roberta-base", is_test=False)
    collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, vocab_size=tok.vocab_size, is_test=False)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    doc_len_threshold = 800
    counter0 = Counter() # shorter than threshold
    counter1 = Counter() # longer than threshold
    with tqdm(total=len(train_dataloader.dataset)) as progress_bar:
        for batch in train_dataloader:
            src_input_lens = batch["src_input_lens"]
            src_keywords_inclusion_types = batch["src_keywords_inclusion_types"]
            counter0.update(src_keywords_inclusion_types[src_input_lens <= doc_len_threshold].tolist())
            counter1.update(src_keywords_inclusion_types[src_input_lens > doc_len_threshold].tolist())
            
            progress_bar.update(src_input_lens.size(0))

    propensity_scores = OrderedDict()    
    for key in sorted(set(counter0.keys()) | set(counter1.keys())):
        propensity_scores[key] = counter1[key] / (counter0[key] + counter1[key])
    with open('save/propensity_scores.json', 'w') as f:
        dump(propensity_scores, f)

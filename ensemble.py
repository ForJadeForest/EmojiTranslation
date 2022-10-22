import math
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer


def get_score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    predictions = bertMaskedLM(tensor_input).logits
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(predictions.squeeze(), tensor_input.squeeze()).data
    return math.exp(loss)


use_ppl = False
submit_path = Path('submit/ensemble_14')
best_name = 'bart-large-beam-search.csv'
data = {}
df = pd.read_csv('raw_data/emoji7w-test_data.csv', sep='\t')
for i, p in enumerate(submit_path.iterdir()):
    if p.name.startswith('reduce'):
        continue
    if p.suffix == '.csv':
        print(p)
        data[p.name] = pd.read_csv(p, sep='\t')
        df[p.name] = data[p.name]['prediction']
        # if p.name.startswith('reduce'):
        #     df[p.name + '_2'] = data[p.name]['prediction']

single_num = 0
res = []
bertMaskedLM = BertForMaskedLM.from_pretrained('Bert-Fine-tune/checkpoint-27200')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
if use_ppl:
    for i, row in tqdm(df.iloc[:, 2:].iterrows(), total=len(df)):
        vote = row.value_counts()
        if len(vote) == 1:
            res.append(vote.index[0])
        elif vote[0] == vote[1] or vote[0] == vote[1] + 1:
            idx = 0
            while idx + 1 < len(vote) and vote[idx + 1] >= vote[0] - 1:
                idx += 1
            min_score = None
            target = None
            scores = {}
            for s in vote.index[: idx + 1]:
                score = get_score(s) / vote[s]
                scores[s] = score
                if min_score is None or score < min_score:
                    target = s
                    min_score = score
            res.append(target)
        else:
            res.append(vote.index[0])
else:
    for i, row in tqdm(df.iloc[:, 2:].iterrows(), total=len(df)):
        vote = row.value_counts()
        if len(vote) == 1:
            res.append(vote.index[0])
        elif vote[0] == vote[1]:
            if vote[data[best_name].prediction[i]] == vote[0]:
                res.append(data[best_name].prediction[i])
            else:
                idx = 0
                while idx + 1 < len(vote) and vote[idx + 1] == vote[0]:
                    idx += 1
                min_score = None
                target = None
                scores = {}
                for s in vote.index[: idx + 1]:
                    score = get_score(s) / vote[s]
                    scores[s] = score
                    if min_score is None or score < min_score:
                        target = s
                        min_score = score
                res.append(target)
        else:
            res.append(vote.index[0])
print(f'single_vote_num: {single_num}')

sentences = pd.read_csv('./raw_data/emoji7w-test_data.csv', sep='\t')
sentences.prediction = pd.DataFrame(res)
sentences.to_csv(submit_path / 'reduced_no_ppl.csv', index=False, sep='\t')

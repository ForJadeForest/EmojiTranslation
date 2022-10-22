import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class Config:
    model_checkpoint = "uer/bart-large-chinese-cluecorpussmall"
    emoji_path = Path('raw_data/all_emoji_large.bin')  # 保存token的文件路径
    max_input_length = 100  # 最大输入长度
    max_target_length = 100  # 最大输出长度
    lr = 5e-6
    weight_decay = 1e-2
    epoch = 50
    pred_batch_size = 64  # test dataset batch_size
    batch_size = 64  # train and val dataset batch_size
    val_size = 0.1  # 验证集百分比
    warmup_ratio = 0.09
    pretrained = True
    ex_name = "bart-large-50epoch"  # 上传wandb的名称，也是本地checkpoint保存名称
    save_path = 'submit_{}.csv'.format(ex_name)  # 结果保存路径
    debug = False
    label_smooth = 0


def postprocess_text(preds, labels):
    preds = [pred.split(' ') for pred in preds]
    labels = [[label.split(' ')] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # pred : [batch_size, max_lengt]: idx of token

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, )

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    wer_score = 0
    bleu_score = 0
    for l, p in zip(decoded_labels, decoded_preds):
        bleu_score += 50 ** sentence_bleu(l, p, weights=(0, 1, 0, 0))
        wer_score += 50 ** (1 - wer(' '.join(l[0]), ' '.join(p)))
    score = bleu_score + wer_score
    bleu_score /= len(decoded_labels)
    wer_score /= len(decoded_labels)
    score /= len(decoded_labels)

    result = {
        "score": score,
        'bleu_score': bleu_score,
        'wer_score': wer_score
    }

    return result


def preprocess_function(examples):
    inputs = [ex for ex in examples["src"]]
    targets = [ex for ex in examples["tgt"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def generate_translation(batch):
    inputs = tokenizer(batch["prediction"], padding="max_length", max_length=max_input_length, truncation=True,
                       return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    outputs = model.generate(input_ids,
                             eos_token_id=102,
                             num_beams=4,
                             do_sample=True,
                             )
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["pred_translation"] = output_str

    return batch


if __name__ == '__main__':
    data = pd.read_excel('./raw_data/emoji7w.xlsx').iloc[:, :6]
    data.columns = ['id', 'src_dirty', 'src', 'is_delete', 'common_emoji', 'tgt']
    data = data.set_index('id')
    if Config.debug:
        data = data.iloc[:200, :]
    data['src'] = data['src'].apply(lambda x: x.replace(' ', '[BLANK]'))
    data['tgt'] = data['tgt'].apply(lambda x: x.replace(' ', '[BLANK]'))
    dataset = Dataset.from_pandas(data.loc[:, ['src', 'tgt']]).train_test_split(Config.val_size)

    tokenizer = AutoTokenizer.from_pretrained(Config.model_checkpoint, use_fast=False, do_lower_case=False)

    emoji_path = Config.emoji_path
    print(len(tokenizer))
    if emoji_path.exists():
        with open(emoji_path, 'br') as f:
            emoji_path = pickle.load(f)
        for emoji in tqdm(emoji_path):
            if emoji not in tokenizer.get_vocab():
                tokenizer.add_tokens(emoji)
        print(len(tokenizer))
    else:
        need_add_num = 0
        emoji_list = []
        bar = tqdm(data['src'], desc='len(emoji_list)=X, len(tokenzier)=X')
        for sentence in bar:
            for char in sentence:
                if char not in tokenizer.get_vocab():
                    tokenizer.add_tokens(char)
                    need_add_num += 1
                    emoji_list.append(char)
                    assert len(set(emoji_list)) == len(emoji_list), print(emoji_list)
                    bar.set_description(f'len(emoji_list)={len(emoji_list)}, len(tokenzier)={len(tokenizer)}')
                    # bar.write(char)

        with open(emoji_path, 'wb') as f:
            pickle.dump(emoji_list, f)
        print(len(tokenizer))

    tokenizer.add_tokens('[BLANK]')
    tokenizer.add_tokens('”')
    tokenizer.add_tokens('“')
    tokenizer.add_tokens('‘')
    tokenizer.add_tokens('’')

    max_input_length = Config.max_input_length
    max_target_length = Config.max_target_length
    source_lang = "src"
    target_lang = "tgt"
    if Config.pretrained:
        model = AutoModelForSeq2SeqLM.from_pretrained(Config.model_checkpoint)
    else:
        model = AutoModelForSeq2SeqLM.from_config(config=AutoConfig.from_pretrained(Config.model_checkpoint))
    model.resize_token_embeddings(len(tokenizer))
    model.config.update({'eos_token_id': 102,
                         'max_length': max_input_length,
                         # 'top_k': 30,
                         # 'top_p': 0.95
                         })

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, max_length=max_input_length)

    args = Seq2SeqTrainingArguments(
        Config.ex_name,
        evaluation_strategy="epoch",
        learning_rate=Config.lr,
        per_device_train_batch_size=Config.batch_size,
        per_device_eval_batch_size=Config.batch_size,
        weight_decay=Config.weight_decay,
        save_total_limit=3,  # 至多保存模型个数
        num_train_epochs=Config.epoch,

        predict_with_generate=True,
        fp16=False,
        run_name=Config.ex_name,
        warmup_ratio=Config.warmup_ratio,
        metric_for_best_model='score',
        load_best_model_at_end=True,
        greater_is_better=True,
        save_strategy='epoch',
        dataloader_num_workers=16,
        label_smoothing_factor=Config.label_smooth,
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,

    )

    trainer.train()
    model.eval()
    test = pd.read_csv('./raw_data/emoji7w-test_data.csv', sep='\t')
    test['prediction'] = test['prediction'].apply(lambda x: x.replace(' ', '[BLANK]'))
    test_data = Dataset.from_pandas(test)

    batch_size = Config.pred_batch_size  # change to 64 for full evaluation
    results = test_data.map(generate_translation, batched=True, batch_size=batch_size)
    sentences = pd.read_csv('./raw_data/emoji7w-test_data.csv', sep='\t')
    sentences.prediction = pd.DataFrame([r.replace(' ', '') for r in results['pred_translation']])
    sentences.prediction = sentences.prediction.apply(lambda x: x.replace('[BLANK]', ' '))
    sentences.to_csv(Config.save_path, index=False, sep='\t')

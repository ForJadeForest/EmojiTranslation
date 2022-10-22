import datetime
import pickle
import re
import time

import numpy as np
import pandas as pd
import transformers
from datasets import Dataset
from emoji import is_emoji
from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, BertTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

import wandb


def gene_sentence(model, tokenizer, save_path, gene_para=None, max_input_length=100, debug=False):
    if gene_para is None:
        gene_para = dict(num_beams=10, do_sample=True)

    def generate_translation(batch):
        inputs = tokenizer(batch["prediction"], padding="max_length", max_length=max_input_length, truncation=True,
                           return_tensors="pt")

        input_ids = inputs.input_ids
        input_ids = input_ids.to('cuda')
        outputs = model.generate(input_ids,
                                 eos_token_id=102,
                                 **gene_para
                                 )
        output_str = tokenizer.batch_decode(outputs)
        batch["pred_translation"] = output_str

        return batch

    test = pd.read_csv('./raw_data/emoji7w-test_data.csv', sep='\t')
    test['prediction'] = test['prediction'].apply(lambda x: x.replace(' ', '[BLANK]'))
    if debug:
        test = test.iloc[:20, :]
    test_data = Dataset.from_pandas(test)
    if model.device.type == 'cpu':
        model.to('cuda')
    model.eval()

    batch_size = 64  # change to 64 for full evaluation
    start_time = datetime.datetime.now()
    results = test_data.map(generate_translation, batched=True, batch_size=batch_size)
    end_time = datetime.datetime.now()

    print(f'prediction time waste: {end_time - start_time} s')
    sentences = pd.read_csv('./raw_data/emoji7w-test_data.csv', sep='\t')
    if debug:
        sentences = sentences.iloc[:1, :]
    sentences.prediction = pd.DataFrame([r.replace(' ', '') for r in results['pred_translation']])
    sentences.prediction = sentences.prediction.apply(lambda x: x.replace('[BLANK]', ' ')) \
        .apply(lambda x: x.replace('[CLS]', '').replace('[PAD]', '').replace('[SEP]', ''))
    sentences.to_csv(save_path, index=False, sep='\t')

    labels = pd.read_csv('./raw_data/emoji7w-test_data.csv', sep='\t')
    try:
        filter_sentence(sentences, labels, str(save_path).replace('.csv', '_filter.csv'))
    except:
        print('Filter Fail. ')


def filter_sentence(pred, labels, save_path):
    res = []
    for _p, _s in zip(pred['prediction'], labels['prediction']):
        if '[UNK]' in _p:
            print(_p, _s)
            new_p, new_s = change(_p, _s)
            print(new_p, new_s)
            print('=' * 20)
            res_p = new_p
        else:
            res_p = _p
        res.append(res_p)
    sentences = pd.read_csv('./raw_data/emoji7w-test_data.csv', sep='\t')
    sentences.prediction = pd.DataFrame(res)
    sentences.to_csv(save_path, index=False, sep='\t')


def delete_one_unk(p, s):
    special = '️'  # special是emoji中的特殊字符
    p = p.replace('[UNK][UNK]', '[UNK]')  # 把连续的unk转化为单个unk
    pos = [substr.start() for substr in re.finditer('\[UNK]', p)]
    i = pos[0]

    if i == 0:
        find_char = p[i + 5]  # 找[unk]下一个字符
        s_idx = [substr.start() for substr in re.finditer(find_char, s)]
        if s_idx:
            sub_char = s[:s_idx[0]]
            new_p = p.replace('[UNK]', sub_char, 1)
        else:
            print('!!!!!!, fail!')
            new_p = p.replace('[UNK]', '', 1)

    elif i == len(p) - 5:
        find_char = p[i - 1]
        s_idx = [substr.start() for substr in re.finditer(find_char, s)]
        if len(s_idx):
            sub_char = s[s_idx[-1] + 1:]
            if is_emoji(sub_char):
                new_p = p.replace('[UNK]', '', 1)
            else:
                new_p = p.replace('[UNK]', sub_char, 1)
        else:
            new_p = p.replace('[UNK]', '', 1)
    else:
        find_char = re.compile(re.escape(p[i - 1]))

        s_idx = [substr.start() for substr in re.finditer(find_char, s)]
        if s_idx:
            if len(s_idx) > 1:
                print('more char in s', p, s)
            s_point = s_idx[0] + 1
            p_point = i + 5
            # print(s_point, p_point)
            while s[s_point] != p[p_point]:
                if is_emoji(s[s_point]):
                    break
                s_point += 1

            sub_char = s[s_idx[0] + 1: s_point]
            new_p = p.replace('[UNK]', sub_char, 1)
        else:
            print('!!!!!!, fail!')
            new_p = p.replace('[UNK]', '', 1)
    new_p = new_p.replace(special, '')
    return new_p, s


def change(p, s):
    pos = [substr.start() for substr in re.finditer('\[UNK]', p)]
    while len(pos) >= 1:
        p, s = delete_one_unk(p, s)
        pos = [substr.start() for substr in re.finditer('\[UNK]', p)]
    return p, s


def add_unk_token(tokenizer, data, emoji_path):
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

        with open(emoji_path, 'wb') as f:
            pickle.dump(emoji_list, f)
        print(len(tokenizer))
    tokenizer.add_tokens('[BLANK]')
    tokenizer.add_tokens('”')
    tokenizer.add_tokens('“')
    tokenizer.add_tokens('‘')
    tokenizer.add_tokens('’')
    return tokenizer


def blank_change(df):
    df['src'] = df['src'].apply(lambda x: x.replace(' ', '[BLANK]'))
    df['tgt'] = df['tgt'].apply(lambda x: x.replace(' ', '[BLANK]'))
    return df


def preprocess_function(examples, tokenizer, max_input_length, max_target_length):
    inputs = [ex for ex in examples["src"]]
    targets = [ex for ex in examples["tgt"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding='max_length')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.split(' ') for pred in preds]
    labels = [[label.split(' ')] for label in labels]

    return preds, labels


def k_fold_train(config, train_data, val_data):
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

    tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint, use_fast=False, do_lower_case=False)
    all_data = pd.concat([val_data, train_data])
    tokenizer = add_unk_token(tokenizer, all_data, config.emoji_path)

    train_df = blank_change(train_data)
    val_df = blank_change(val_data)
    if config.debug:
        train_df = train_df.iloc[:200, :]
        val_df = val_df.iloc[:20, :]
        config.epoch = 3
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    wandb.init(project="Emoji-Huggingface",
               name=config.ex_name,
               group=config.group)
    max_input_length = config.max_input_length

    if config.pretrained:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_checkpoint)
    else:
        model = AutoModelForSeq2SeqLM.from_config(config=AutoConfig.from_pretrained(config.model_checkpoint))
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.config.update({'eos_token_id': 102,
                         'max_length': max_input_length,
                         'use_cache': False
                         })

    fn_kwargs = {
        'tokenizer': tokenizer,
        'max_input_length': config.max_input_length,
        'max_target_length': config.max_input_length

    }
    train_tokenized_datasets = train_dataset.map(preprocess_function, batched=True, fn_kwargs=fn_kwargs)
    val_tokenize_datasets = val_dataset.map(preprocess_function, batched=True, fn_kwargs=fn_kwargs)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, max_length=max_input_length,
                                           padding='max_length')

    args = Seq2SeqTrainingArguments(
        config.checkpoint_path,
        evaluation_strategy="epoch",
        learning_rate=config.lr,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        save_total_limit=3,  # 至多保存模型个数
        num_train_epochs=config.epoch,

        predict_with_generate=True,
        fp16=config.fp16,
        run_name=config.ex_name,
        warmup_ratio=config.warmup_ratio,
        metric_for_best_model='score',
        load_best_model_at_end=True,
        greater_is_better=True,
        save_strategy='epoch',
        dataloader_num_workers=config.num_worker,
        label_smoothing_factor=config.label_smooth,
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=val_tokenize_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    gene_sentence(model,
                  tokenizer,
                  save_path=config.save_path,
                  debug=config.debug,
                  gene_para=config.gene_para
                  )
    wandb.finish()


def train_pl(config, train_data, val_data):
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

    print('[INFO] ==> load pl data')
    pl_src = pd.read_csv('raw_data/emoji7w-test_data.csv', sep='\t')
    pl_tgt = pd.read_csv(config.pl_path, sep='\t')
    pl_tgt.columns = ['id', 'tgt']
    pl_src.columns = ['id', 'src']

    pl_data = pl_src.merge(pl_tgt)
    pl_data = pl_data.set_index('id')

    pl_all_data = pd.concat([pl_data, train_data])
    train_data = blank_change(train_data)
    pl_all_data = blank_change(pl_all_data)
    val_data = blank_change(val_data)

    if config.debug:
        print('[INFO] ==> Now is debug model')
        train_data = train_data.iloc[:200, :]
        pl_all_data = pl_all_data.iloc[:200, :]
        val_data = val_data.iloc[:20, :]
        config.pl_epoch = 2
        config.epoch = 2

    pl_dataset = Dataset.from_pandas(pl_all_data.loc[:, ['src', 'tgt']])
    train_dataset = Dataset.from_pandas(train_data.loc[:, ['src', 'tgt']])
    val_dataset = Dataset.from_pandas(val_data.loc[:, ['src', 'tgt']])
    if config.model_checkpoint.startswith('fnlp'):
        tokenizer = BertTokenizer.from_pretrained(config.model_checkpoint, use_fast=False, do_lower_case=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint, use_fast=False, do_lower_case=False)

    wandb.init(project="Emoji-Huggingface",
               name=config.ex_name,
               group=config.group)
    emoji_path = config.emoji_path
    all_data = pd.concat([val_data, train_data])
    tokenizer = add_unk_token(tokenizer, all_data, emoji_path)

    max_input_length = config.max_input_length
    print('[INFO] ==> begin load model')
    if config.pretrained:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_checkpoint)
    else:
        model = AutoModelForSeq2SeqLM.from_config(config=AutoConfig.from_pretrained(config.model_checkpoint))
    print('[INFO] ==> load model done')
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.resize_token_embeddings(len(tokenizer))
    model.config.update({'eos_token_id': 102,
                         'max_length': max_input_length,
                         })

    fn_kwargs = {
        'tokenizer': tokenizer,
        'max_input_length': config.max_input_length,
        'max_target_length': config.max_input_length

    }

    print('[INFO] ==> begin tokenizer data')
    tokenized_datasets = train_dataset.map(preprocess_function, batched=True, fn_kwargs=fn_kwargs)
    pl_tokenized_datasets = pl_dataset.map(preprocess_function, batched=True, fn_kwargs=fn_kwargs)
    val_dataset_tokenized_datasets = val_dataset.map(preprocess_function, batched=True, fn_kwargs=fn_kwargs)
    print('[INFO] ==> tokenizer data done')
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, max_length=max_input_length, padding='max_length')

    args = Seq2SeqTrainingArguments(
        config.checkpoint_path / 'pl_step',
        evaluation_strategy="epoch",
        learning_rate=config.lr,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        num_train_epochs=config.pl_epoch,
        predict_with_generate=True,
        fp16=config.fp16,
        run_name=config.ex_name,
        warmup_ratio=config.warmup_ratio,
        metric_for_best_model='score',
        load_best_model_at_end=True,
        greater_is_better=True,
        save_strategy='epoch',
        dataloader_num_workers=config.num_worker,
        label_smoothing_factor=config.label_smooth,
    )

    opt = transformers.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=pl_tokenized_datasets,
        eval_dataset=val_dataset_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(opt, None)
    )

    trainer.train()

    del trainer, args

    args = Seq2SeqTrainingArguments(
        config.checkpoint_path / 'norm_step',
        evaluation_strategy="epoch",
        learning_rate=config.lr,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        num_train_epochs=config.epoch,
        predict_with_generate=True,
        fp16=config.fp16,
        run_name=config.ex_name,
        warmup_ratio=config.warmup_ratio,
        metric_for_best_model='score',
        load_best_model_at_end=True,
        greater_is_better=True,
        save_strategy='epoch',
        dataloader_num_workers=config.num_worker,
        label_smoothing_factor=config.label_smooth,
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets,
        eval_dataset=val_dataset_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,

    )
    trainer.train()

    gene_sentence(model,
                  tokenizer,
                  save_path=config.save_path,
                  debug=config.debug
                  )
    wandb.finish()

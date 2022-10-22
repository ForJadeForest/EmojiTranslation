"""
This file is to finetune bert-base model. For calculating perplexity to compare 2 sentence.
"""
import pandas as pd
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, BertTokenizer, BertForMaskedLM, TrainingArguments, Trainer
import os

data = pd.read_excel('./raw_data/emoji7w.xlsx').iloc[:, :6]
data.columns = ['id', 'src_dirty', 'src', 'is_delete', 'common_emoji', 'tgt']

dataset = Dataset.from_pandas(data.loc[:, ['tgt']]).train_test_split(0.01)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', )
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
model = BertForMaskedLM.from_pretrained('bert-base-chinese')
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def tokenize_function(examples):
    result = tokenizer(examples["tgt"], padding='max_length', max_length=100, truncation=True)
    return result


block_size = 128


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    result["labels"] = result["input_ids"].copy()
    return result


dataset = dataset.map(tokenize_function, batched=True)
training_args = TrainingArguments(
    output_dir="Bert-Fine-tune",
    evaluation_strategy="epoch",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    warmup_ratio=0.1,
    dataloader_num_workers=16,
    save_strategy='epoch',
    num_train_epochs=50,
    metric_for_best_model='loss',
    greater_is_better=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()
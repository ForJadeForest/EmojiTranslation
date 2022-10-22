import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

cpk_path = '/home/pyz/code/Emoji-Translation_Huggingface/bart-base-p-labelnorm_step/checkpoint-14835'
save_path = 'bart-base-pl-beam.csv'
model = AutoModelForSeq2SeqLM.from_pretrained(cpk_path)
tokenizer = AutoTokenizer.from_pretrained(cpk_path, use_fast=False)
max_input_length = 128

test = pd.read_csv('./raw_data/emoji7w-test_data.csv', sep='\t')
test['prediction'] = test['prediction'].apply(lambda x: x.replace(' ', '[BLANK]'))
test_data = Dataset.from_pandas(test)
model.to('cuda:3')
model.eval()


def generate_translation(batch):
    inputs = tokenizer(batch["prediction"], padding="max_length", max_length=max_input_length, truncation=True,
                       return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda:3")
    outputs = model.generate(input_ids,
                             eos_token_id=102,
                             num_beams=4,
                             do_sample=True,

                             )
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["pred_translation"] = output_str

    return batch


batch_size = 64  # change to 64 for full evaluation
results = test_data.map(generate_translation, batched=True, batch_size=batch_size)
sentences = pd.read_csv('./raw_data/emoji7w-test_data.csv', sep='\t')
sentences.prediction = pd.DataFrame([r.replace(' ', '') for r in results['pred_translation']])
sentences.prediction = sentences.prediction.apply(lambda x: x.replace('[BLANK]', ' '))
sentences.to_csv(save_path, index=False, sep='\t')

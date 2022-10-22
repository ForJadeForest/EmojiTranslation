"""
This file is use model to generation result and use post process.
"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils import gene_sentence

cpk_path = r'/home/pyz/code/Emoji-Translation_Huggingface/bart-large-p-labelpl_step/checkpoint-5865'
save_path = r'./test_speed_sub.csv'
device = 'cuda:1'

model = AutoModelForSeq2SeqLM.from_pretrained(cpk_path)
tokenizer = AutoTokenizer.from_pretrained(cpk_path, use_fast=False, do_lower_case=False)

gene_sentence(model, tokenizer, save_path, max_input_length=100,
              gene_para=dict(num_beams=4, do_sample=True),
              debug=True)

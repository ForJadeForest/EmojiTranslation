import os
from pathlib import Path

import pandas as pd

from utils import k_fold_train


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class Config:
    def __init__(self):
        self.debug = False
        self.device = 'cuda:1'
        self.label_smooth = 0.3

        self.ex_name = "bart-large_5e-6_smooth0.3"  # 上传wandb的名称，也是本地checkpoint保存名称
        self.lr = 5e-6
        self.epoch = 50

        self.model_checkpoint = "uer/bart-large-chinese-cluecorpussmall"
        # self.model_checkpoint = 'fnlp/bart-large-chinese'
        self.emoji_path = Path('raw_data/all_emoji_large.bin')  # 保存token的文件路径

        self.max_input_length = 128  # 最大输入长度
        self.max_target_length = 128  # 最大输出长度

        self.weight_decay = 1e-2
        self.pred_batch_size = 64  # test dataset batch_size
        self.batch_size = 64  # train and val dataset batch_size
        self.warmup_ratio = 0.1
        self.pretrained = True

        self.save_base_path = Path('/data/pyz')
        self.checkpoint_path = self.save_base_path / 'checkpoint' / self.ex_name
        self.save_path = self.save_base_path / 'checkpoint' / self.ex_name

        self.seed = 2022
        self.num_worker = 0
        self.kfold = 5
        self.group = self.ex_name
        self.pl_path = Path('raw_data/reduced_ppl+bestvote_0.87302.csv')
        self.gradient_checkpointing = True
        self.fp16 = True
        self.gene_para = dict(
            num_beams=10,
            do_sample=False,
        )


if __name__ == '__main__':
    config = Config()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.device[-1]
    for i in range(config.kfold):
        print(f'[INFO] ==> Begin fold {i}')
        config.ex_name = config.ex_name + f'_fold_{i}'
        config.checkpoint_path = config.checkpoint_path / f'fold_{i}'
        config.save_path = config.save_path / f'submit_fold_{i}.csv'
        train_df = pd.read_csv(f'./raw_data/train_fold_{i}.csv', sep='\t')
        val_df = pd.read_csv(f'./raw_data/val_fold_{i}.csv', sep='\t')
        k_fold_train(config, train_df, val_df)
        print(f'[INFO] ==> fold {i} is done')
        config = Config()

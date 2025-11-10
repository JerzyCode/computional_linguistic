import os

import sentencepiece as spm

PREPARED_DATASET_DIR = "./prepared_data"

spm.SentencePieceTrainer.train(
    input=os.path.join(PREPARED_DATASET_DIR, "train.txt"),
    model_prefix="sentencepiece",
    vocab_size=52000,
    model_type="unigram",
    character_coverage=1.0,
    pad_id=0,
    bos_id=2,
    eos_id=3,
    unk_id=1,
)

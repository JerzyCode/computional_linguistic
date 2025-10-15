from datetime import datetime, timedelta

import torch
from transformers import AutoTokenizer

from lstm.data import load_data, load_data_subset
from lstm.lstm_model import Lstm
from lstm.training import train
from lstm.utils import EpochTracker, ModelSaver

TRAINING_DATA_PATH = "lab1/datasets/training_data_small.pt"
EVAL_DATA_PATH = "lab1/datasets/eval_data_small.pt"
METRICS_CSV_PATH = "metrics/metrics.csv"
SAVE_MODEL_DIR = "models/"
BATCH_SIZE = 32
EVAL_SUBSET = 3

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


tokenizer = AutoTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
model_saver = ModelSaver(SAVE_MODEL_DIR, save_frequency=5)
epoch_tracker = EpochTracker(METRICS_CSV_PATH)
lstm_model = Lstm(device=device, tokenizer=tokenizer, seq_len=128, dtype=torch.float16)

end_training_date = datetime.now() + timedelta(minutes=5)


if __name__ == "__main__":
    train_dataset = load_data(TRAINING_DATA_PATH, device, batch_size=BATCH_SIZE)
    eval_dataset = load_data_subset(
        EVAL_DATA_PATH, device, batch_size=BATCH_SIZE, subset_size=EVAL_SUBSET
    )

    train(
        model=lstm_model,
        training_set=train_dataset,
        evaluation_set=eval_dataset,
        end_training_date=end_training_date,
        epoch_tracker=epoch_tracker,
        model_saver=model_saver,
        device=device,
    )

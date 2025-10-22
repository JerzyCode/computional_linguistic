from datetime import datetime, timedelta

import torch
from transformers import AutoTokenizer

from lstm.data import load_data_subset
from lstm.logger import log
from lstm.lstm_model import Lstm
from lstm.training import train
from lstm.utils import EpochTracker, ModelSaver

TRAINING_DATA_PATH = "lab1/datasets/training_data_small.pt"
EVAL_DATA_PATH = "lab1/datasets/eval_data_small.pt"
METRICS_CSV_PATH = "metrics/metrics.csv"
SAVE_MODEL_DIR = "models/"
BATCH_SIZE = 10
EVAL_SUBSET = 5

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
log.info(f"Device: {device}")


tokenizer = AutoTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
model_saver = ModelSaver(SAVE_MODEL_DIR, save_frequency=15)
epoch_tracker = EpochTracker(METRICS_CSV_PATH)

end_training_date = datetime.now() + timedelta(minutes=5)


if __name__ == "__main__":
    lstm_model = Lstm(
        device=device,
        tokenizer=tokenizer,
        hidden_size=128,
        emb_dim=128,
        dtype=torch.float32,
    )

    log.info(f"Model parameters count: {lstm_model.count_lstm_parameters()}")

    train_dataset = load_data_subset(
        TRAINING_DATA_PATH,
        device,
        batch_size=BATCH_SIZE,
        subset_size=5,
    )
    eval_dataset = load_data_subset(
        EVAL_DATA_PATH,
        device,
        batch_size=BATCH_SIZE,
        subset_size=EVAL_SUBSET,
    )
    first_batch = next(iter(train_dataset))
    X, y = first_batch

    # Wypisz kilka tokenów
    num_examples_to_show = min(5, X.shape[0])
    num_tokens_to_show = min(10, X.shape[1])  # ile tokenów w sekwencji

    for i in range(num_examples_to_show):
        print(f"Example {i}:")
        print("  X:", X[i, :num_tokens_to_show].tolist())
        print("  y:", y[i, :num_tokens_to_show].tolist())

    train(
        model=lstm_model,
        training_set=train_dataset,
        evaluation_set=eval_dataset,
        end_training_date=end_training_date,
        epoch_tracker=epoch_tracker,
        model_saver=model_saver,
        device=device,
        lr=5e-3,
        log_batch_freq=10,
        use_amp=True,
    )

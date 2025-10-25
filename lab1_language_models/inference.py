import time

import torch
import torch.nn as nn
from lstm import Lstm
from torchmetrics.text.perplexity import Perplexity
from training import create_data_loader, tokenizer

torch.manual_seed(42)


SEQ_LEN = 128
MAX_EVAL_ELEMENTS = 1000
MAX_TRAIN_ELEMENTS = 1000
device = "cpu"


prompts = [
    "",
    "Jesteś potwornie brzydkie — rzekły dzikie kaczki",
    "Nazajutrz w kąciku pod murem, ujrzano zmarznięte ciało dziewczynki",
    "nie ufając na ślepo nowinkom i wiedząc z doświadczenia, że nie zawsze gorączka rozwoju",
    "Chcemy mówić o kunszcie poetyckim samym w",
]


def load_model(path: str, model: nn.Module):
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {path}")
    return model


def evaluate_perplexity_torchmetrics(
    model,
    eval_dataloader,
    train_dataloader,
):
    model.to(device)
    model.eval()
    start_time = time.time()

    eval_metric = Perplexity()
    train_metric = Perplexity()

    batch_count = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(train_dataloader, 1):
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)
            train_metric.update(logits, y)
            batch_count += 1

            print(f"Train Batch {batch_idx}/{len(eval_dataloader)} processed")

        for batch_idx, (x, y) in enumerate(eval_dataloader, 1):
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)
            eval_metric.update(logits, y)
            batch_count += 1

            print(f"Eval Batch {batch_idx}/{len(eval_dataloader)} processed")

    eval_pp = eval_metric.compute()
    train_pp = train_metric.compute()
    total_time = time.time() - start_time
    return eval_pp.item(), train_pp.item(), total_time / batch_count, batch_count


if __name__ == "__main__":
    eval_dataset = create_data_loader(
        path="prepared_data/val_data.pt",
        seq_len=SEQ_LEN,
        max_elements=MAX_EVAL_ELEMENTS,
    )

    train_dataset = create_data_loader(
        path="prepared_data/train_data.pt",
        seq_len=SEQ_LEN,
        max_elements=MAX_EVAL_ELEMENTS,
    )

    # model = TransformerLanguageModel(
    #     vocab_size=tokenizer.vocab_size,
    #     embedding_dim=512,
    #     num_heads=8,
    #     blocks_count=8,
    #     seq_len=SEQ_LEN,
    #     device=device,
    #     dropout=0.3,
    # ).to(device)

    # non_trained_model = TransformerLanguageModel(
    #     vocab_size=tokenizer.vocab_size,
    #     embedding_dim=512,
    #     num_heads=8,
    #     blocks_count=8,
    #     seq_len=SEQ_LEN,
    #     device=device,
    #     dropout=0.3,
    # ).to(device)

    model = Lstm(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=552,
        layers_count=8,
        hidden_size=552,
        seq_len=SEQ_LEN,
        dropout=0.3,
        device=device,
    ).to(device)

    non_trained_model = Lstm(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=552,
        layers_count=8,
        hidden_size=552,
        seq_len=SEQ_LEN,
        dropout=0.3,
        device=device,
    ).to(device)

    model = load_model("final_models/lstm/lstm_30_20251025_183501.pt", model)

    eval_pp, train_pp, batch_time, batches = evaluate_perplexity_torchmetrics(
        model, eval_dataset, train_dataset
    )
    print("TRAINED MODEL RESULTS:")
    print(f"Eval Perplexity: {eval_pp:.6f}")
    print(f"Train Perplexity: {train_pp:.6f}")
    print(f"Time: {batch_time:.2f}s for {batches} batches, seq_len:{SEQ_LEN}")

    eval_pp, train_pp, batch_time, batches = evaluate_perplexity_torchmetrics(
        non_trained_model, eval_dataset, train_dataset
    )

    print("UNTRAINED MODEL RESULTS:")
    print(f"Eval Perplexity: {eval_pp:.6f}")
    print(f"Train Perplexity: {train_pp:.6f}")
    print(f"Time: {batch_time:.2f}s for {batches} batches, seq_len:{SEQ_LEN}")

    for i, prompt in enumerate(prompts):
        print(f"\nPROMPT: {i + 1}")
        print(f"{prompt}")

        completion = model.generate_text(tokenizer, text=prompt, max_tokens=128)
        completion = completion.replace("[CLS]", "").replace("[SEP]", "").strip()

        print("\nCompletion:")
        print(f"{completion}")

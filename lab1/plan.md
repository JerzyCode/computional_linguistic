## Task: Create RRN model and Transformers model

### Plan:

1) Plan the experiment
2) Select dataset, create some prompts. # Load dataset and tokenizer
3) Remind about RNN's - LSTM
4) Implement LSTM
5) Train LSTM and save
6) Examine Trained LSTM Model
7) Read attention is all you need
8) Read more about transformers
9) Implement Transformer
10) Train Transformer and save
11) Examine Transformer
12) Make report comparing both models

13) Maybe dragon hatching?

### 1. Plan

Task:  causal language task - predict next token

Find nice dataset and tokenizer


Implement firstly in jupyter both models, and then copy it to .py files to run it virtualy.

Files:
- lstm.py - lstm implementation
- transformer.py - transformer implementation
- train.py - training function for models
- utils.py - tracking metrics, saving models, loading dataset
- logger.py 

Evaluation during training should be done on different set.

Tracking metrics is important. Both models should be trained the same duration.

After training each model, load it locally and test :) 

Then make a repot (maybe latex) about results and implementation, problems etc.


### 2. Dataset

Wolne_lektury - treningowy
1000_novels_corpus_CLARIN-PL - ewaluacyjny
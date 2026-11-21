# Neural Machine Translation with Attention (English → German)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shirsteiner/neural-machine-translation-attention/blob/main/nmt-seq2seq-attention.ipynb)

A PyTorch sequence-to-sequence model that translates English image captions into German. I built a GRU encoder–decoder, extended it with additive (Bahdanau) attention, then analyzed where the model goes wrong and what its attention learns.

**→ [View the notebook](nmt-seq2seq-attention.ipynb)** (all outputs and plots are saved, so there's no need to run it)

## Results

| Model | Test BLEU |
|---|---|
| GRU encoder–decoder | 14.0 |
| GRU encoder–decoder + additive attention | **24.2** |

Both models were trained on [Multi30k](https://github.com/multi30k/dataset) (~29k sentence pairs) for 20k steps on a T4 GPU. BLEU is NLTK corpus BLEU on lowercased text, so it isn't directly comparable to published sacreBLEU scores.

## Key findings

- **Attention adds ~10 BLEU.** The decoder can look back at the relevant source words at every step instead of squeezing the whole sentence into one vector.
- **Most errors are plausible substitutions**, such as *hat → vest*, *fixing → riding* and *snowmobiles → umbrellas*. The attention maps point to two different causes: sometimes the model attends to a neighboring word (*orange* instead of *hat*), and sometimes it attends to the right word but picks the wrong translation (*fixing → fahren*).
- **Attention behaves differently for content words and grammar words.** Concrete nouns and verbs get almost all of the attention on a single source word (*gebäude ← building* 0.95, *schneidet ← cutting* 0.97). German articles spread their attention across the noun phrase, because the right form depends on the noun's gender and case.

## What's in the notebook

1. **Data preprocessing:** tokenization, BPE subwords, vocabularies
2. **Basic encoder–decoder:** GRU encoder and decoder, masked cross-entropy loss, training loop with validation BLEU
3. **Attention:** additive attention layer and an attentive seq2seq model
4. **Analysis:** error analysis, attention heatmaps and patterns in high-attention token pairs

## My contribution

This project started from a university NLP course template, which provided the data pipeline, model skeletons, BLEU evaluation and plotting helpers. I implemented the decoder step, the training loss, the training loop, the attention layer, the attentive model and attention extraction (marked `### MY CODE` in the notebook), trained both models and wrote all of the analysis.

## Running it

Open the notebook in Colab with a GPU runtime and run all cells. It installs its dependencies and downloads the data itself. Training takes about 30 minutes on a T4 GPU (10 for the basic model, 21 with attention), and results vary slightly from run to run.

**Stack:** Python, PyTorch, subword-nmt (BPE), NLTK, pandas, matplotlib

## Acknowledgements

- Data: Multi30k (Elliott et al., 2016), via the [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k) Hugging Face mirror
- Attention: Bahdanau, Cho & Bengio (2015), [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473)
- Diagrams in the notebook: [Lena Voita, NLP Course For You](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)

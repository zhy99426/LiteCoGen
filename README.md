# LiteCoGen: Lightweight Collaborative Posterior Correction for Generative Recommendation


## Environment

```bash
pip install -r requirements.txt
```

Python 3.9+, CUDA recommended.

## Datasets

Experiments use Amazon Reviews 2018 from UCSD:
Toys_and_Games, Sports_and_Outdoors, Movies_and_TV.

### Preprocessing

```bash
python data/preprocess.py --dataset Toys_and_Games --output_dir ./data/Toys_and_Games
python data/preprocess.py --dataset Sports_and_Outdoors --output_dir ./data/Sports_and_Outdoors
python data/preprocess.py --dataset Movies_and_TV --output_dir ./data/Movies_and_TV
```

### Item Embeddings

```bash
python data/generate_embeddings.py --dataset Toys_and_Games --data_dir ./data --device cuda:0
python data/generate_embeddings.py --dataset Sports_and_Outdoors --data_dir ./data --device cuda:0
python data/generate_embeddings.py --dataset Movies_and_TV --data_dir ./data --device cuda:0
```

Uses `huggyllama/llama-7b` with mean pooling over title, description, and features.

## How to Run

Train LiteCoGen:

```bash
python run.py --config config.yaml --data_path ./data --dataset Toys_and_Games --output_dir ./output --device cuda:0 --model litecogen --stage all --seed 42
```


Pipeline stages (`--stage`): `prepare` | `semid` | `train` | `all`.

## Repository Structure

```
LiteCoGen/
├── README.md
├── requirements.txt
├── config.yaml
├── run.py
├── data/
│   ├── preprocess.py
│   └── generate_embeddings.py
└── src/
    ├── litecogen.py
    ├── tiger.py
    ├── t5.py
    ├── trie.py
    ├── rqvae.py
    ├── dataset.py
    ├── metrics.py
    ├── trainer.py
    ├── utils.py
    └── __init__.py
```

## License

MIT License

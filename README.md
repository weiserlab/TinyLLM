# TinyLLM
A Framework for training, fine-tuning and deploying smaller LLMs on custom datasets.

Website:
ArXiv:

## Introduction

## Installation

Setting up:
- setup llm.c (GPU setup required). Find instructions [here]()
- setup llama.cpp [here]
- install requirements (FT, datasets processing)

## Usage

### Preparing the pre-training datasets
- cd into datasets folder
```bash
cd Datasets/
```
- use ```encode.py``` to tokenise different datasets. You can use your own custom data (csv format) or those hosted on HuggingFace. To begin with, the scripts use Fineweb (available on HuggingFace, and 10 Billion tokens variant is automatically downloaded when the script runs) and SHL (an IoT sensor dataset, instructions for downloading [here]()).
```bash
python encode.py
```
- rename the tokensied datasets appropriately (```Fineweb```/ ```SHL```)
- use ```split.py``` to split the tokens proportionally and prepare master pretraining dataset
```
python split.py -d1 0.3 -d2 0.7 -o ./pretraining_data
```
- tweak the parameters in ```split.py``` if required. Currently set to produce 9BT, Training:Val dataset tokens in 98:2 ratio, the output shards (datafiles) are 100MB each
- If facing with storage, memory issues, try reducing the shard size from 100MB to 75MB or even less


### Pre-training the model

1) Navigate to the llm.c folder
```
cd ../llm.c/
```

2) Begin pre-training, by appropriately setting the parameters
```
./train_gpt2cu \
    -i "Datasets/pretraining_data/train*.bin" \
    -j "Datasets/pretraiing_data/val*.bin" \
    -o "custom_model" \
    -e "d12" \
    -b 64 -t 1024 \
    -d 524288 \
    -r 1 \
    -z 1 \
    -c 0.1 \
    -l 0.0006 \
    -q 0.0 \
    -u 700 \
    -n 10000 \
    -v 250 -s 20000 \
    -h 1
```
Parameters:
- e: Parameters of the model (d6, d8, d10, d11, d12) <Enter e vs parameters>
- o: Output directory of the trained model
More info on the parameters can be found here: []()
(steps for resuming the training)

3) Export the model into huggingface compatible format (optional)
```bash
lf=$(ls custom_model/model_000*.bin | sort -V | tail -n 1) # select latest model
python dev/eval/export_hf.py -i "$lf" -o "custom_model_hf"
```
, here lf is the name of the model produced at the end of the training.


### Fine-tuning the model

(copy the original tokeniser files to the fine-tuning folder)

### Deploying the model
- convert the model to GGUF
- quantisation (optional)

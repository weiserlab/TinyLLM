# TinyLLM  
*A Framework for Training, Fine-Tuning, and Deploying Smaller LLMs on Custom Datasets*

- **Website**: [TinyLLM.org](https://tinyllm.org/)  
- **ArXiv**: Coming up 

---

## Introduction  
TinyLLM is a lightweight and customizable framework designed to enable efficient training, fine-tuning, and deployment of small-scale Large Language Models (LLMs) on custom datasets. It is optimized for resource-constrained environments, making it ideal for edge and IoT applications.

---

## Installation  

### Prerequisites  
1. **Dependency Setup**:  
   - Configure `llm.c` (GPU setup required). Follow the instructions [here](https://github.com/karpathy/llm.c/discussions/481).  
   - Set up `llama.cpp`. Refer to the build guide [here](https://github.com/VIS-WA/llama.cpp/blob/master/docs/build.md).  
   
2. **Additional Requirements**:  
   - Install necessary Python dependencies for dataset processing and fine-tuning.  
   ```bash
   pip install -r requirements.txt
   ```
   - Install suitable version of [PyTorch](https://pytorch.org/get-started/locally/)

---

## Usage  

### 1. Preparing Pre-training Datasets  
<details>
<summary>Click to expand</summary>

1. Navigate to the datasets folder:  
   ```bash
   cd Datasets/
   ```

2. Tokenize datasets using `encode.py`:  
   - Supports user's private custom data (in CSV format) or datasets hosted on HuggingFace.  
   - By default, the script processes Fineweb (10 Billion tokens variant, auto-downloaded) and SHL (IoT sensor dataset).
   - Download SHL dataset following the instructions [here](https://github.com/weiserlab/TinyLLM/tree/main/Datasets/SHL).
   - Change the `datasets_to_tokenize` parameter in `encode.py` appropriately to tokenise custom datasets (if any) to include them later in the pre-training process.  
   ```bash
   python encode.py
   ```

3. Rename tokenized datasets for clarity:  
   - Example: `Fineweb`, `SHL`.

4. Split datasets using `split.py`:  
   ```bash
   python split.py -d1 0.3 -d2 0.7 -o ./pretraining_data
   ```
   - Current defaults produce a dataset with 9 Billion tokens, with Training:Validation split in a 98:2 ratio with 100MB shards.
   - Adjust parameters if needed. Change the `dataset1_path` and `dataset1_path` to include other datasets. More than 2 datasets can be used too by simple modifications to the script.  
   - Note that the minimum number of tokens in training/ validation datasets would be in multiples of number of tokens present in a 100MB shard (~52M tokens). This means that the actual split ratio may vary slightly from the input ratio if the `TOTAL_TOKENS` is less i.e. ~5B range. The `CHUNK_SIZE` in `encode.py` can be reduced in such cases to get a more accurate split ratio.
   - If faced with memory constraints, reduce shard size from 100MB to 75MB or smaller.  
</details>

---

### 2. Pre-training the Model  

<details>
<summary>Click to expand</summary>
   
1. Navigate to the `llm.c` folder:  
   ```bash
   cd ../llm.c/
   ```

2. Begin pre-training with the following command:  
   ```bash
   ./train_gpt2cu \
       -i "Datasets/pretraining_data/train*.bin" \
       -j "Datasets/pretraining_data/val*.bin" \
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

3. Flags::  
   - `-e`: Model depth (e.g., `d6`, `d8`, `d10`, `d12`). [Add parameter details here].  
   - `-o`: Output directory for the trained model.
   - use `-y 1` to resume from a last checkpoint (in the same input directory).
   - By default, the training happens for 1 epoch. To train for more epochs, use the `-x` flag.
   - For a full list of flags and descriptions, refer [here](https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L1369) and [here](https://github.com/karpathy/llm.c/discussions/481).  

4. Export the model in HuggingFace-compatible format:  
   ```bash
   lf=$(ls custom_model/model_0*.bin | sort -V | tail -n 1) # Select the latest model
   python dev/eval/export_hf.py -i "$lf" -o "custom_model_hf"
   ```
</details>

---

### 3. Fine-tuning the Model 

<details>
<summary>Click to expand</summary>

1. Navigate to the `Fine-tune` folder:  
   ```bash
   cd ../Fine-tune/
   ```
2. Set the required parameters for fine-tuning in the corresponding model's parameter file (`p-{model}.txt`). Currently 3 models are supported: GPT-2 (custom), Llama-3 and Phi-3.

3. Run the fine-tuning script:  
   ```bash
    python master.py \
    -d "breathe" \
    -m "../llm.c/custom_model_hf" \
    -n "gpt2" \
    -p "p-gpt.txt" | tee ft_output.log
   ```
   - `-d`: Dataset name (e.g., `breathe`, `local`, `swim` or `gesture`).
   - `-m`: Path to the pre-trained model.
   - `-n`: Model name (e.g., `gpt2`, `llama`, `phi`).
   - `-p`: Parameter file for the model. (Check and chage this)

- View the loss plot directly in the terminal or in the `results/{model}/{dataset}/` directory. To view in the terminal,

```bash
cat results/{model}/{dataset}/loss.txt
```
- Note that `test.csv` has a copy of the same testing data 5 or 10 times to simplify the accuracy readings later.
- Currently, 3 in-house processed datasets (gesture detection, localisation and breathing detection) are provided for fine-tuning, apart from `swim` which has to processed (find more details [here](https://github.com/weiserlab/TinyLLM/blob/main/Datasets/swim/process.ipynb)). More information about the datasets will be updated soon [here](www.huggingface.co/tinyllm).

- Note that the checkpoints created as part of fine-tuning process can be removed later to save space.


</details>

---

### 4. Deploying and Inferencing the Model  

<details>
<summary>Click to expand</summary>

1. Once, a fine-tuned model is created, we can use this model directly with HuggingFace's transformers library
```python3
from transformers import pipeline
import torch
 
# Path to the local fine-tuned model directory
path = "./TinyLLM/Fine-tune/results/GPT 2/breathe-0/"
prompt = "The sea is blue but it's his red sea"
 
generator = pipeline("text-generation", model=path,max_new_tokens = 30, repetition_penalty=1.3, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
print(generator(prompt)[0]['generated_text'])
```
1. For usage on embedded devices, or other CPU devices, it might be useful to convert the model to GGUF format (using llama.cpp).  
```bash
cd ../llama.cpp/ #TinyLLM/llama.cpp/
python convert_hf_to_gguf.py "../Fine-tune/results/GPT 2/breathe-0/"  --outfile "../Fine-tune/results/GPT 2/breathe-0/model.gguf"
```
2. This model can be now used with llama.cpp as follows:
```bash
./llama-cli -m "../Fine-tune/results/GPT 2/breathe-0/model.gguf" -n 10 -p "### Instruction:\nYou have an array of data capturing the frequency of chest movements of a human being, recorded at a sampling rate of 31.8 Hz over a duration of approximately 30 seconds. The data ranges from 0 to 100. Using this data, calculate the person's breathing rate in breaths per minute.\n\n### Input:\n44 36 41 46 15 26 33 38 31 44 38 44 41 36 31 28 59 56 31 54 64 64 67 64 59 72 69 67 72 72 82 77 82 92 82 85 85 87 95 90 79 77 90 87 87 87 92 95 82 82 82 87 74 72 77 56 62 64 64 64 62 54 41 49 46 44 46 36 44 44 44 44 46 41 33 26 33 38 33 36 13 41 38 33 38 44 44 46 36 36 33 38 44 33 33 44 49 54 54 46 56 51 62 59 67 56 59 72 74 51 64 72 74 72 77 77 74 77 69 77 74 77 72 72 64 77 77 72 67 79 74 79 67 67 64 56 56 56 62 64 56 49 44 51 59 51 46 44 44 41 49 33 33 41 41 51 51 46 41 41 49 49 41 41 44 33 41 41 41 36 36 51 41 38 46 44 28 54 44 44 41 46 41 41 41 46 54 51 49 51 69 67 62 64 67 64 62 74 72 74 69 74 72 72 87 77 79 77 79 85 85 77 85 79 85 77 90 90 74 79 79 85 85 74 56 62 59 59 64 56 64 62 59 51 54 51 51 46 49 36 44 46 44 44 38 33 26 41 41 41 33 23 33 33 0 36 33 28 31 23 26 28 31 28 44 49 41 23 33 33 38 28 28 44 36 33 38 31 41 49 38 44 59 59 69 46 46 69 69 62 54 74 82 72 69 87 69 87 92 79 92 87 74 77 77 77 92 77 82 92 90 90 92 92 85 90 59 67 82 69 77 77 72 56 67 54 54 49 54 46 51 56 41 46 51 51 44 44 36 38 36 33 33 36 31 36 26 26 26 36 33 21 13 15 13 8 13 5 26 5 13 18 10 18 18 10 28 26 15 23 18 28 21 28 36 38 31 44 38 51 62 59 59 56 54 54 54 62 59 64 67 72 82 72 82 82 82 79 85 82 85 87 72 90 82 87 85 92 87 90 95 90 92 90 85 90 87 74 69 74 79 69 72 72 64 56 56 62 64 54 44 49 54 41 62 41 41 31 33 44 36 28 26 21 31 28 36 31 28 33 21 28 18 33 31 18 15 33 31 21 23 21 23 23 23 31 31 31 33 31 33 41 51 46 26 33 36\n\n### Response:" # 10.5
```
2. Optional: Quantize the model for optimized inference. Read more [here](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)  

</details>

---

## Contributing  
We welcome contributions to TinyLLM!

Also find collection of models pre-trained on web and sensor data on our [HuggingFace](https://huggingface.co/TinyLLM) page. 

---

## Acknowledgments  
Thank you to the creators of [llm.c](https://github.com/karpathy/llm.c) and [llama.cpp](https://github.com/ggerganov/llama.cpp) for making LLM training and usage more accessible.  

We also acknowledge the use of the following external datasets in this framework:  
- [Fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)  
- [SHL](https://www.shl-dataset.org/)  
- [ExtraSensory](http://extrasensory.ucsd.edu/)  

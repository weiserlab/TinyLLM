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
   Install necessary Python dependencies for dataset processing and fine-tuning.  
   ```bash
   pip install -r requirements.txt
   ```

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
   - Supports custom data in CSV format or datasets hosted on HuggingFace.  
   - By default, the script processes Fineweb (10 Billion tokens variant, auto-downloaded) and SHL (IoT sensor dataset, download instructions [here]()).  
   ```bash
   python encode.py
   ```

3. Rename tokenized datasets for clarity:  
   - Example: `Fineweb`, `SHL`.

4. Split datasets using `split.py`:  
   ```bash
   python split.py -d1 0.3 -d2 0.7 -o ./pretraining_data
   ```
   - Adjust parameters if needed. Current defaults produce a dataset with 9 Billion tokens, with Training:Validation split in a 98:2 ratio with 100MB shards.  

5. Optimize for storage/memory:  
   - If necessary, reduce shard size from 100MB to 75MB or smaller.  
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

3. Parameters:  
   - `-e`: Model depth (e.g., `d6`, `d8`, `d10`, `d12`). [Add parameter details here].  
   - `-o`: Output directory for the trained model.
   - use `-y 1` to resume from a last checkpoint (in the same input directory).
   - For a full list of parameters and descriptions, refer [here](https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L1369) and [here](https://github.com/karpathy/llm.c/discussions/481).  

4. Resuming training: [Add steps here]  

5. Optional: Export the model in HuggingFace-compatible format:  
   ```bash
   lf=$(ls custom_model/model_000*.bin | sort -V | tail -n 1) # Select the latest model
   python dev/eval/export_hf.py -i "$lf" -o "custom_model_hf"
   ```
</details>

---

### 3. Fine-tuning the Model 

<details>
<summary>Click to expand</summary>

- Copy the original tokenizer files into the fine-tuning directory.  
- [Add detailed fine-tuning steps here].  

</details>

---

### 4. Deploying the Model  

<details>
<summary>Click to expand</summary>

1. Convert the model to GGUF format.  
2. Optional: Quantize the model for optimized inference.  
3. [Add deployment steps here].

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

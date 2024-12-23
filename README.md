# TinyLLM  
*A Framework for Training, Fine-Tuning, and Deploying Smaller LLMs on Custom Datasets*

- **Website**: [TinyLLM.org](https://tinyllm.org/)  
- **ArXiv**: [ArXiv/2412.15304](https://arxiv.org/abs/2412.15304)



## Introduction  
TinyLLM is a lightweight and customizable framework for efficiently training, fine-tuning, and deploying small-scale Large Language Models (LLMs) on custom datasets. It is optimized for resource-constrained environments, making it ideal for applications on edge devices and IoT platforms.

TinyLLM demonstrates adaptability across various datasets, particularly in embedded sensing tasks like hand gesture detection, robot localization, and breathing rate detection. The framework enables training smaller models for diverse domains and is not limited to embedded sensing.



## Installation  

### Prerequisites  
1. **Dependency Setup**:  
   - Configure `llm.c` (GPU setup required). Follow the instructions [here](https://github.com/karpathy/llm.c/discussions/481).  
   - Set up `llama.cpp`. Refer to the build guide [here](https://github.com/VIS-WA/llama.cpp/blob/master/docs/build.md).  

2. **Python Dependencies**:  
   - Install necessary Python libraries for dataset processing and fine-tuning:  
     ```bash
     pip install -r requirements.txt
     ```
   - Install a suitable version of [PyTorch](https://pytorch.org/get-started/locally/).  



## Usage  

### 1. Preparing Pre-training Datasets  
<details>
<summary>Click to expand</summary>

1. Navigate to the datasets folder:  
   ```bash
   cd Datasets/
   ```

2. Tokenize datasets using `encode.py`:  
   - Supports user-provided custom datasets (in CSV format) or datasets hosted on HuggingFace.  
   - By default, the script processes the Fineweb dataset (10 billion tokens variant, auto-downloaded) and the SHL IoT sensor dataset.  
   - Follow the instructions [here](https://github.com/weiserlab/TinyLLM/tree/main/Datasets/SHL) to download the SHL dataset.  
   - Update the `datasets_to_tokenize` parameter in `encode.py` for custom datasets.  
     ```bash
     python encode.py
     ```

3. Rename tokenized datasets for clarity (e.g., `Fineweb`, `SHL`).

4. Split datasets using `split.py`:  
   ```bash
   python split.py -d1 0.3 -d2 0.7 -o ./pretraining_data
   ```
   - Default parameters produce a dataset with 9 billion tokens and a Training:Validation split of 98:2 with 100MB shards.  
   - Adjust parameters or shard size if faced with memory constraints.  

</details>



### 2. Pre-training the Model  

<details>
<summary>Click to expand</summary>
   
1. Navigate to the `llm.c` folder:  
   ```bash
   cd ../llm.c/
   ```

2. Begin pre-training with:  
   ```bash
   ./train_gpt2cu \
       -i "Datasets/pretraining_data/train*.bin" \
       -j "Datasets/pretraining_data/val*.bin" \
       -o "custom_model" \
       -e "d6" \
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

3. Key Flags:  
   - `-e`: Model depth (e.g., `d6`, `d12`).  
   - `-o`: Output directory for the trained model.  
   - `-y 1`: Resume from the last checkpoint.  
   - Use `-x` for multiple epochs.  
   - Full list of flags and descriptions is [here](https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L1369).  

4. Export the model in HuggingFace-compatible format:  
   ```bash
   lf=$(ls custom_model/model_0*.bin | sort -V | tail -n 1) 
   python dev/eval/export_hf.py -i "$lf" -o "custom_model_hf"
   ```
</details>



### 3. Fine-tuning the Model

<details>
<summary>Click to expand</summary>

1. Navigate to the `Fine-tune` folder:  
   ```bash
   cd ../Fine-tune/
   ```

2. Set parameters in the respective model's parameter file (e.g., `p-gpt.txt`).  

3. Run the fine-tuning script:  
   ```bash
   python master.py \
       -d "breathe" \
       -m "../llm.c/custom_model_hf" \
       -n "gpt2" \
       -p "p-gpt.txt" | tee ft_output.log
   ```
   - `-d`: Dataset name (e.g., `breathe`, `gesture`).  
   - `-m`: Path to the pre-trained model.  
   - `-n`: Model name (`gpt2`, `llama`, `phi`).  

4. Results:  
   - Training and evaluation loss plots are saved in `results/{model}/{dataset}/loss.pdf`. To view in the terminal,
      ```bash
         cat "results/GPT 2/breathe-0/loss.txt"  
      ```
   - Testing data can be viewed directly or processed for analysis.  

</details>



### 4. Inferencing the Model  

<details>
<summary>Click to expand</summary>

1. Use HuggingFace's transformers library for inference:  
   ```python
   from transformers import pipeline
   import torch
   
   path = "./TinyLLM/Fine-tune/results/GPT 2/breathe-0/"
   generator = pipeline("text-generation", model=path, max_new_tokens=30, repetition_penalty=1.3, device_map="auto")
   prompt = "Your input text here"
   print(generator(prompt)[0]['generated_text'])
   ```

2. Convert the model to GGUF format for embedded devices:  
   ```bash
   cd ../llama.cpp/
   python convert_hf_to_gguf.py "../Fine-tune/results/GPT 2/breathe-0/" --outfile "../Fine-tune/results/GPT 2/breathe-0/model.gguf"
   ```

3. Use the model with llama.cpp:  
   ```bash
   ./llama-cli -m "../Fine-tune/results/GPT 2/breathe-0/model.gguf" -n 10 -p "Your input prompt"
   ```

4. Optionally, quantize the model for optimized inference ([details](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)).  

</details>


### Notes
- Currently, 3 in-house processed datasets (gesture detection, localisation, and breathing detection) are provided for fine-tuning, apart from `swim`, which has to be processed (find more about using the dataset [here](https://github.com/weiserlab/TinyLLM/blob/main/Datasets/swim/process.ipynb)). More information about the in-house datasets will be updated soon [here](www.huggingface.co/tinyllm).
- The checkpoints created during the fine-tuning process can be removed later to save space.

## Contributing  
We welcome contributions to TinyLLM! Visit our [HuggingFace](https://huggingface.co/TinyLLM) page for pre-trained models on web and sensor data.  



## Acknowledgments  
Thank you to the creators of [llm.c](https://github.com/karpathy/llm.c) and [llama.cpp](https://github.com/ggerganov/llama.cpp) for their groundbreaking tools.  

We also acknowledge the use of external datasets:  
- [Fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)  
- [SHL](https://www.shl-dataset.org/)  
- [ExtraSensory](http://extrasensory.ucsd.edu/)  

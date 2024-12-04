import os
from random import randrange
from functools import partial
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                        #   BitsAndBytesConfig,
                        #   HfArgumentParser,
                        #   Trainer,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                        #   EarlyStoppingCallback,
                        #   logging,
                          set_seed)
from transformers.pipelines.pt_utils import KeyDataset
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM
import pandas as pd
#from accelerate import Accelerator
from trl import SFTTrainer
import matplotlib.pyplot as plt
import plotext as pltt
from tqdm.auto import tqdm

instruct = ""
intro = ""
#intro = "Below is an instruction that provides sensor data. Write a response that appropriately completes the request. You will be tipped for answering correctly."

def create_prompt_formats(sample):

    global intro
    # Initialize static strings for the prompt template
    # INTRO_BLURB = ""
    # INTRO_BLURB = "Below is an instruction that provides sensor data. Write a response that appropriately completes the request. You will be tipped for answering correctly."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "### Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"

    # Combine a prompt with the static strings
    blurb = f"{intro}"
    if 'instruction' in sample:
        instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
    else:
        instruction = f"{INSTRUCTION_KEY}\n{instruct}"
    input_context = f"{INPUT_KEY}\n{sample['input']}" if sample["input"] else None
    response = f"{RESPONSE_KEY}\n{sample['output']}"
    end = f"{END_KEY}"

    # Create a list of prompt template elements
    parts = [part for part in [blurb, instruction, input_context, response, end] if part]
    # Join prompt template elements into a single string to create the prompt template
    formatted_prompt = "\n\n".join(parts)
    # Store the formatted prompt template in a new key "text"
    sample["text"] = formatted_prompt

    return sample



def get_max_length(model):

    # Initialize a "max_length" variable to store maximum sequence length as null
    max_length = None
    # Find maximum sequence length in the model configuration and save it in "max_length" if found
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    # Set "max_length" to 1024 (default value) if maximum sequence length is not found in the model configuration
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    print(f"Max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):

    return tokenizer(
        batch["text"],
        max_length = max_length,
        truncation = True,
    )

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str, name: str):

    # Add prompt to each sample
    print("Preprocessing " + name + " dataset...")
    dataset = dataset.map(create_prompt_formats)

    # Apply preprocessing to each batch of the dataset & and remove "instruction", "input", "output", and "text" fields
    _preprocessing_function = partial(preprocess_batch, max_length = max_length, tokenizer = tokenizer)
    columns = dataset.column_names

    dataset = dataset.map(
        _preprocessing_function,
        batched = True,
        remove_columns = columns,
    )
    # check if there are samples with "input_ids" exceeding "max_length"
    if len(dataset.filter(lambda sample: len(sample['input_ids']) > max_length)):

        print(f"Number of samples with input_ids exceeding max_length: {len(dataset.filter(lambda sample: len(sample['input_ids']) > max_length))}")
        # Filter out samples that have "input_ids" exceeding "max_length"
        dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    # dataset = dataset.shuffle(seed = seed)

    # print the size of the dataset
    print(f"{name} Dataset size: {len(dataset)}")

    return dataset

def load_model_dataset(model_name, model_dir, device_t, bnb_config, training_dataset_name, validation_dataset_name, seed, instruction):

    print("Loading Model "+model_name+"...\n") 
    n_gpus = torch.cuda.device_count()
    max_memory = torch.cuda.get_device_properties(0).total_memory/1024**2
    print("Max Memory (GB)",max_memory/1024)
    max_memory=f'{max_memory}MB'
    #max_memory = f'{40960}MB'

    print("Present working Directory",os.getcwd())
    print(f"Number of GPUs: {n_gpus}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config = bnb_config,
        device_map = device_t,
        torch_dtype = torch.bfloat16,
        max_memory = {i: max_memory for i in range(n_gpus)},
        trust_remote_code=True,

    )
    print("Memory Loaded")
        # attn_implementation="flash_attention_2",
        #device_map = {"":0},
        #max_memory = {i: max_memory for i in range(n_gpus)},
        #low_cpu_mem_usage=True,
        #load_in_8bit_fp32_cpu_offload=True

    # Load model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir,add_eos_token=True,    trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    if model_name == "Microsoft Phi 3":
        tokenizer.padding_side = 'right' # to prevent warnings Gemma
    else:
        tokenizer.padding_side = 'right' # left
    print(f'\n***{tokenizer.padding_side} Padded***\n')
    # print("Max length of the model:",tokenizer.model_max_length)

    # Load training dataset
    if validation_dataset_name != "":
        dataset = load_dataset("csv", data_files = training_dataset_name, split = "train")
    # Load validation dataset
        validation_dataset = load_dataset("csv", data_files = validation_dataset_name, split = "train")
    else:
        whole_dataset  = load_dataset(training_dataset_name,split = "train")
        train_test_split = whole_dataset.train_test_split(test_size=0.1)
        train_valid_split = train_test_split['train'].train_test_split(test_size=0.0526)  # 5% of 95% is about 0.0526
        	    

	# Access the train, validation, and test splits
        dataset = train_valid_split['train']
        validation_dataset = train_valid_split['test']
        # test_dataset = train_test_split['test']



    print(f'Number of prompts: {len(dataset)}')
    print(f'Column names are: {dataset.column_names}')

    max_length = get_max_length(model)
    print(f"Max length of the model: {max_length}")

    global instruct
    instruct = instruction

    training_preprocessed_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset,"Training")
    validation_preprocessed_dataset = preprocess_dataset(tokenizer, max_length, seed, validation_dataset,"Validation")

    return model, tokenizer, training_preprocessed_dataset, validation_preprocessed_dataset, max_length


def find_all_linear_names(model):

    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit = False):

    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
    )

def fine_tune(model_dir,
          model,
          tokenizer,
          dataset,
          validation_dataset,
          lora_r,
          lora_alpha,
          lora_dropout,
          bias,
          task_type,
          per_device_train_batch_size,
          gradient_accumulation_steps,
          warmup_steps,
          max_steps,
          learning_rate,
          fp16,
          logging_steps,
          output_dir,
          optim,
          max_length,
          device_t,
          output_merged_dir,
          save_parameters = True):

    os.makedirs(output_dir, exist_ok=True)

    # Enable gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # Prepare the model for training
    model = prepare_model_for_kbit_training(model)

    # Get LoRA module names
    target_modules = find_all_linear_names(model)

    # if model 
    # modules_to_save = ["lm_head", "embed_token"]

    # Create PEFT configuration for these modules and wrap the model to PEFT
    peft_config = LoraConfig(
        r = lora_r,
        lora_alpha = lora_alpha,
        target_modules = target_modules,
        lora_dropout = lora_dropout,
	# modules_to_save = modules_to_save,
        bias = bias,
        task_type = task_type,
    )
    pms = get_peft_model(model, peft_config)
    #print(model)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(pms)
    args = TrainingArguments(
            per_device_train_batch_size = per_device_train_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            warmup_steps = warmup_steps,
            max_steps = max_steps,
            learning_rate = learning_rate,
            fp16 = fp16,
            logging_steps = logging_steps,
            output_dir = output_dir,
            optim = optim,
            do_eval=True,
            eval_strategy="steps",
            eval_steps=logging_steps,
            save_steps=logging_steps,
            eval_accumulation_steps=1,
            save_total_limit=1,
            # load_best_model_at_end=True,
            # metric_for_best_model="eval_loss",
            # greater_is_better=False,
            report_to="none",            
        )
    # Training parameters
    # trainer = Trainer(
    #     model = model,
    #     train_dataset = dataset,
    #     eval_dataset = validation_dataset,
    #     args = TrainingArguments(
    #         per_device_train_batch_size = per_device_train_batch_size,
    #         gradient_accumulation_steps = gradient_accumulation_steps,
    #         warmup_steps = warmup_steps,
    #         max_steps = max_steps,
    #         learning_rate = learning_rate,
    #         fp16 = fp16,
    #         logging_steps = logging_steps,
    #         output_dir = output_dir,
    #         optim = optim,
    #         do_eval=True,
    #         evaluation_strategy="steps",
    #         eval_steps=logging_steps,
    #         save_steps=logging_steps,
    #         save_total_limit=1,
    #         load_best_model_at_end=True,
    #         metric_for_best_model="eval_loss",
    #         greater_is_better=False,
    #         report_to="none",            
    #     ),
    #     data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
    # )
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset = validation_dataset,
        args = args,
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False),
        peft_config = peft_config,
        dataset_text_field = "text",
        max_seq_length = max_length,
        # packing = True,
    )

    model.config.use_cache = False

    do_train = True

    # Launch training and log metrics
    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

    # Save model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok = True)
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model, trainer
    torch.cuda.empty_cache()

    # Load fine-tuned weights
    # model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map = device_t, torch_dtype = torch.bfloat16, load_in_8bit=load_in_8bit,low_cpu_mem_usage=True) 
    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map = device_t, torch_dtype = torch.bfloat16,low_cpu_mem_usage=True,trust_remote_code=True)
    # Merge the LoRA layers with the base model
    model = model.merge_and_unload()

    # Save fine-tuned model at a new location
    # output_merged_dir = output_dir + "final_merged_checkpoint"
    os.makedirs(output_merged_dir, exist_ok = True)
    model.save_pretrained(output_merged_dir, safe_serialization = True)

    # Save the tokenizer at the same location
    tokenizer.save_pretrained(output_merged_dir)

    if save_parameters:
        with open(output_merged_dir + "/parameters.txt", "w") as file:
            file.write(f"lora_r: {lora_r}\n")
            file.write(f"lora_alpha: {lora_alpha}\n")
            file.write(f"lora_dropout: {lora_dropout}\n")
            file.write(f"bias: {bias}\n")
            file.write(f"task_type: {task_type}\n")
            file.write(f"per_device_train_batch_size: {per_device_train_batch_size}\n")
            file.write(f"gradient_accumulation_steps: {gradient_accumulation_steps}\n")
            file.write(f"warmup_steps: {warmup_steps}\n")
            file.write(f"max_steps: {max_steps}\n")
            file.write(f"learning_rate: {learning_rate}\n")
            file.write(f"fp16: {fp16}\n")
            file.write(f"logging_steps: {logging_steps}\n")
            file.write(f"optim: {optim}\n")
            file.write(f"output_dir: {output_dir}\n")
            # file.write(f"load_in_8bit: {load_in_8bit}\n")

        file.close()   
    print("***Clearing Memory***")
    del model, tokenizer
    torch.cuda.empty_cache()

    # copy the original model files except *.safetensors to the merged directory
    # replace if the file already exists

    for file in os.listdir(model_dir):
        if not file.endswith(".safetensors"):
            os.system(f'cp -f {model_dir}/{file} "{output_merged_dir}"{file}')

 
def loss_plot(path,out_dir):
    print("***Plotting the loss***")
    with open(path, 'r') as file:
        data = file.read()
        data = data.split('\n')
        eval_loss = [float(line.split('eval_loss')[1].split(',')[0].split(': ')[1]) for line in data if 'eval_loss' in line]
        loss = [float(line.split("'loss'")[1].split(',')[0].split(': ')[1]) for line in data if "'loss'" in line]

        plt.plot(loss, label="loss")
        plt.plot(eval_loss, label='eval_loss')
        plt.legend()
        plt.savefig(out_dir+'/loss.pdf', format='pdf')
        plt.plot(loss, label="loss")
        plt.plot(eval_loss, label='eval_loss')

    print("***Plotting the shell plot in loss.txt***")
    pltt.plot(loss, label = 'loss'); pltt.plot(eval_loss, label = 'Eval loss')
    plot_output = pltt.build()
    with open(out_dir+'/loss.txt', 'w') as f:
        f.write(plot_output)


def accuracy(testing_data, out_model_dir, instruction, model_name):
    print(out_model_dir)
    print("***Testing the model***")

    df = pd.read_csv(testing_data,header=0)
    print("The detected columns are:",df.columns)
    tokenizer = AutoTokenizer.from_pretrained(out_model_dir)
    #tokenizer = AutoTokenizer.from_pretrained("./work/results/GPT 2/local-0/")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(out_model_dir)
    #model = AutoModelForCausalLM.from_pretrained("./work/results/GPT 2/local-0/")
    if out_model_dir[-1] == "/":
        out_model_dir = out_model_dir[:-1]

    generated_output = []
    i = 0
    accuracy = 0
    global intro
    with open(out_model_dir + "/accuracies.txt", "w") as file:
        file.write("")

        # tqdm for progress bar in df.input
        for prompt in tqdm(df.input):
            input_prompt = f'{intro}\n\n### Instruction:\n{instruction}\n\n### Input:\n{prompt}\n### Response:\n'

            #sequences = pipe(input_prompt)
            tokens = tokenizer.encode(prompt, return_tensors="pt")
            output = model.generate(tokens, max_new_tokens=30, repetition_penalty=1.3, pad_token_id=tokenizer.pad_token_id)
            #print(f'***Length of output tokens:{len(tokens[0])}')
            output = tokenizer.batch_decode(output)
            out = []
            for o in output:
                out.append(f'{o}')

            #print(f'***Input***\n{input_prompt}')
            #print(f'***The generated output***\n{out}')
            dataset = out_model_dir.split('/')[-1].split('-')[0]
            #print(f'***The generated output***\n{out}')
            try:
                #print(dataset)
                # Define the part of the output to be compared with the expected output
                ## USER-DEFINED
                if dataset == 'gesture':
                    print("****************")
                    print(out)
                    print("------")
                    out = out[0].split(prompt)[1].split('\n')[1]
                elif dataset == 'breathe':
                    out = out[0].split("Response:")[1]
                    out = str(out)
                    print(out)
                else:
                    out = out[0].split('\n')
                    '''
                    print("****************")
                    print(out)
                    print("------")
                    '''
            except:
                print("Couldn't split the output")
           # print(f'***The generated output***\n{out}')
            file.write("\n-------------------------------\n")  
            #continue

            file.write(str(out))
            file.write("\n--------------\n")  

            
            # keep only part after the input prompt
            #out = out[0].split(input_prompt)[1]
            # extract the word by splitting with newline until a word is found
            #out = out.split("\n")
            # iterate through the list and find the first word
            '''
            for word in out: 
                if word: 
                    out = word
                    break
            '''
            generated_output.append(out)

            # MODIFY IF NEEDED
            if str(df.output[i]) in out and out.count(str(df.output[i])) == 1:
                accuracy += 1
            else:
                print(f"{i}: Generated: {generated_output[i]}, Expected: {df.output[i]}")
                file.write(f"{i}: Generated: {generated_output[i]}, Expected: {df.output[i]}\n")
            i += 1

        accuracy = accuracy / len(generated_output)
        print(f"Accuracy: {accuracy * 100}%")
        file.write(f"Accuracy: {accuracy * 100}%")
    file.close()




from fine_tune import load_model_dataset, fine_tune, loss_plot, accuracy
from transformers import BitsAndBytesConfig
import torch
import os
import argparse

home = os.path.abspath("./")
# home->work->dataset/results

# parse arguments from the command line 
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_type", help="Type of dataset: breathe, local, gesture, swim", required=True)
parser.add_argument("-m", "--model_dir", help="Path to the model", required=True)
parser.add_argument("-n", "--model_type", help="Short name of the model: gpt2, llama3-8b, phi3", required=True)
parser.add_argument("-rn", "--random_number", help="Plot log file number, pass job ID", required=False)
parser.add_argument("-p", "--parameter_file", help="File containing the fine_tuning parameters", required=True)
## add model type
args = parser.parse_args()

dataset_type = args.dataset_type
rn = args.random_number
file_path = args.parameter_file
model_type = args.model_type

## load config file and set the parameters
def load_config(file_path):
    with open(file_path, 'r') as file:
        # Execute the config file in the global namespace
        exec(file.read(), globals())

print(dataset_type, model_type)


if model_type == "gpt2":
    model_name = "GPT 2"
    model_dir = args.model_dir
    output_log = f'{home}/gpt2-{dataset_type}-output-{rn}.txt'
elif model_type == "llama3-8b":
    model_name =  "Meta Llama 3 8B"# "Microsoft Phi 3"
    model_dir = args.model_dir
    output_log = f'{home}/llama3-{dataset_type}-output-{rn}.txt'
elif model_type == "phi3":
    model_name = "Microsoft Phi 3"
    model_dir = args.model_dir
    # model_dir = home + 'models/phi3'
    output_log = f'{home}/phi3-{dataset_type}-output-{rn}.txt'
# Add any other models supported by HF's transformers here
else:
    raise ValueError("Model type not recognized")

if dataset_type == "breathe":
    instruction = "You have an array of data capturing the frequency of chest movements of a human being, recorded at a sampling rate of 31.8 Hz over a duration of approximately 30 seconds. The data ranges from 0 to 100. Using this data, calculate the person's breathing rate in breaths per minute."
elif dataset_type == "local":
    instruction = "Sensor data values are provided in the following order: pressure (hPa), humidity (%), temperature (C), humidity (%), Intensity of red, green and blue light, and sound intensity at the place. Using these sensor values, determine where the location of the device. Give your answer only as Meeting Room, Charging Station, and Server Rack."
elif dataset_type == "gesture":
    instruction = "Sensors data values are provided in the following order: proximity, red, green and blue light intensity values. Using these sensor values, determine the hand gesture performed. Give your answer only as Tap, Double and Hold"
elif dataset_type == "swim":
    instruction = "Accelerometer sensor data values are provided along the X, Y, and Z axes. Using these sensor values, determine the swimming style being performed. Your answer should be one of the following: freestyle, breaststroke, backstroke, butterfly, or transition."
elif dataset_type == "online":
    instruction = ""
else:
    raise ValueError("Dataset type not recognized")

output_dir = home + "/results/" + model_name + "/" # model predictions and checkpoints will be stored
output_merged_dir = output_dir + dataset_type

i = 0
while os.path.exists(output_merged_dir + "-" + str(i)):
   i += 1
output_merged_dir = output_merged_dir + "-" + str(i) + "/"
os.makedirs(output_merged_dir, exist_ok=True)
print("Model Source:", model_dir)
print(output_merged_dir)
output_dir = output_merged_dir + "checkpoints/"

# Dataset
if instruction == "": ## Online - make it extendable
    training_dataset_name = "sahil2801/CodeAlpaca-20k" 
    validation_dataset_name = ""
    testing_dataset_name =  ""

else:
    training_dataset_name = home + "datasets/"+ dataset_type + "/train.csv" 
    validation_dataset_name = home + "datasets/" + dataset_type + "/validation.csv"
    testing_dataset_name = home + "datasets/" + dataset_type + "/test.csv" 

# gesture
# LoRA parameters

load_config(file_path)

# TrainingArguments parameters

fp16 = False # True for llama, False for phi # Enable fp16/bf16 training (set bf16 to True with an A100)
load_in_8bit=False # false for llama, False for phi
logging_steps = 1 # Log every X updates steps
seed = 34 # Random seed
save_parameters = True
device_t = "auto" #"auto",cuda:0, {"":Accelerator().process_index}


bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4", # or fp4
        bnb_4bit_compute_dtype = torch.bfloat16,
    )


model, tokenizer, training_preprocessed_dataset, validation_preprocessed_dataset, max_length = load_model_dataset(model_name, model_dir, device_t, bnb_config, training_dataset_name, validation_dataset_name, seed, instruction)

fine_tune(model,tokenizer, training_preprocessed_dataset, validation_preprocessed_dataset, lora_r, lora_alpha, lora_dropout, bias, task_type, per_device_train_batch_size, gradient_accumulation_steps, warmup_steps, max_steps, learning_rate, fp16, logging_steps, output_dir, optim, max_length, device_t, output_merged_dir,save_parameters)

try:
    loss_plot(output_log, output_merged_dir)
except:
    print("Loss not plotted")

accuracy(testing_dataset_name, output_merged_dir, instruction, model_name)
print(output_merged_dir)


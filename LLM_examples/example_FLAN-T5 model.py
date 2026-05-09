# this is simple example of using FLAN-T5 model without any pretraining
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

import pandas as pd
import os
'''
huggingface_ds_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_ds_name)
print("You download a dataset from Hugging Face")


# Save each split as CSV 
# train_dialogsum.csv, validation_dialogsum.csv, test_dialogsum.csv

# Specify save directory
save_dir = r"F:\research\research_learning_code\transformers_llm\dataset"
# Create folder if it does not exist
os.makedirs(save_dir, exist_ok=True)
for split in dataset.keys():
    file_path = os.path.join(save_dir, f"{split}_dialogsum.csv")
    dataset[split].to_csv(file_path)

#--------------------------------------------------------------------------------
#print examples
print("-------------------------------------------------------------------------------")
print("Some example of Dataset")
example_index =[100, 300]
dash_line = '-'.join("" for x in range(100))
for i, index in enumerate(example_index):
    print(dash_line)
    print('Example:', i+1)
    print("input dialog:")
    print(dataset['train']['dialogue'][i])
    print(dash_line)
    print("Human summary:")
    print(dataset['train']['summary'][i])
    print(dash_line)
    print()

#--------------------------------------------------------------------------------
# FLAN-T5 model
model_name ='google/flan-t5-base'    
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
save_directory = r"F:\research\research_learning_code\transformers_llm\model"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

for i, index in enumerate(example_index):
    summary = dataset['test'][index]['summary']
    dialog = dataset['test'][index]['dialogue']

    input_tokens = tokenizer(dialog, return_tensors= 'pt')
    ouput = tokenizer.decode (model.generate(input_tokens["input_ids"], 
    max_new_tokens = 50,)[0], skip_special_tokens = True)

    print(dash_line)
    print('Example:', i+1)
    print("input dialog:")
    print(dataset['test'][index]['dialogue'])
    print(dash_line)
    print("Human summary:")
    print(dataset['test'][index]['summary'])
    print(dash_line)
    print("model generated output:", ouput)
    print()

'''
#-------------------------------------------------------------------------------
#-------------------------------PART 2------------------------------------------
#---------------------- Using the save data and model
load_directory = "./my_local_model"
Ds_Train = pd.read_csv('F:\\research\\research_learning_code\\transformers_llm\\dataset\\train_dialogsum.csv')
Ds_Test = pd.read_csv('F:\\research\\research_learning_code\\transformers_llm\\dataset\\test_dialogsum.csv')
print(Ds_Train.head())
print("size of Training set=", len(Ds_Train))
print("size of Testing set=", len(Ds_Test))

#--------------------------------------------------------------------------------
#print examples
print("-------------------------------------------------------------------------------")
print("Some example of Dataset")
example_index =[100, 300]
dash_line = '-'.join("" for x in range(100))
for i, index in enumerate(example_index):
    print(dash_line)
    print('Example:', i+1)
    print("input dialog:")
    print(Ds_Train['dialogue'][i])
    print(dash_line)
    print("Human summary:")
    print(Ds_Train['summary'][i])
    print(dash_line)
    print()


save_directory = "F:\\research\\research_learning_code\\transformers_llm\\model"
# using the save model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForSeq2SeqLM.from_pretrained(save_directory)
example_index =[100, 300]

for i, index in enumerate(example_index):
    summary = Ds_Test['summary'][i]
    dialog = Ds_Test['dialogue'][i]

    input_tokens = tokenizer(dialog, return_tensors= 'pt')
    ouput = tokenizer.decode (model.generate(input_tokens["input_ids"], 
    max_new_tokens = 50,)[0], skip_special_tokens = True)

    print(dash_line)
    print('Example:', i+1)
    print("input dialog:")
    print(Ds_Test['dialogue'][i])
    print(dash_line)
    print("Human summary:")
    print(Ds_Test['summary'][i])
    print(dash_line)
    print("model generated output:", ouput)
    print()

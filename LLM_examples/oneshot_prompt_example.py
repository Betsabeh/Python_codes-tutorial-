from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

import pandas as pd
import os

#---------------------- Using the save data and model
load_directory = "./my_local_model"
Ds_Train = pd.read_csv('F:\\research\\research_learning_code\\transformers_llm\\dataset\\train_dialogsum.csv')
Ds_Test = pd.read_csv('F:\\research\\research_learning_code\\transformers_llm\\dataset\\test_dialogsum.csv')
print(Ds_Train.head())
print("size of Training set=", len(Ds_Train))
print("size of Testing set=", len(Ds_Test))

#-------------------------------------------------------------------------------
save_directory = "F:\\research\\research_learning_code\\transformers_llm\\model"
# using the save model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForSeq2SeqLM.from_pretrained(save_directory)

tokenizer.tgt_lang ="fa_IR"
print("-------------------------------------------------------------------------------")
#-------------------------------------------------------------------------------
dash_line = '-'.join("" for x in range(100))
prompt =''

Dialogue =Ds_Train['dialogue'][300]
Summery= Ds_Train['summary'][300]
prompt +=f"""
Dialogue:
{Dialogue}
summerization of Dialogue to persian:?
summerize:
{Summery}

"""
Dialogue =Ds_Test['dialogue'][100]
prompt +=f"""
Dialogue:
{Dialogue}
summerization of Dialogue to persian:?
summerize:
"""


# accpet persian language
#model.config.forced.bos_token_id = tokenizer.lang.code_to_id[]

input_token = tokenizer(prompt, return_tensors ='pt')
output = tokenizer.decode (
    model.generate(input_token["input_ids"], max_new_tokens= 50,)
    [0],
    skip_special_tokens = True)

print("Prompt:")
print(prompt)
print("generated translation:")
print(output)
print("orignal summery:")
print(Ds_Test['summary'][100])
    

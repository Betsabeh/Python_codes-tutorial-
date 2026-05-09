from transformers import LlamaForCausalLM, LlamaTokenizer  

# Load the tokenizer and model  
model_name = "your/path/to/llama-model"  # Update with the path to your model  
tokenizer = LlamaTokenizer.from_pretrained(model_name)  
model = LlamaForCausalLM.from_pretrained(model_name)

# Encode input text  
input_text = "What is the future of AI?"  
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)  

# Generate response  
outputs = model.generate(**inputs, max_length=50)  
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  

print(response_text)

import torch

available_cuda = torch.cuda.is_available()
# train a model on the GPU: 
#first send the model itself to the GPU
#it is needed because the trainable parameters of the model need to be on the GPU 
#so that they can be applied and updated in each forward-backward pass
device = torch.device("cuda" if available_cuda else "cpu")
print("Device: ",device)

model = torch.nn.Sequential()
model = model.to(device=device)  

x = torch.randn(3, 4)
print(x)
print(x.device)
x = x.to(device)
print(x.device)

# this code compare the cup and cuda results
import torch  
import torch.nn as nn  
import torch.optim as optim
import time


# Define the DNN model  
class SimpleDNN(nn.Module):  
    def __init__(self, input_size, hidden_size, num_classes):  
        super(SimpleDNN, self).__init__()  
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(hidden_size, num_classes)  
        self.sigmoid = nn.Sigmoid() # Sigmoid for binary classification  

    def forward(self, x):  
        out = self.fc1(x)  
        out = self.relu(out)  
        out = self.fc2(out)  
        out = self.sigmoid(out)  
        return out  

# Hyperparameters  
input_size = 10  
hidden_size = 50  
num_classes = 1  # Binary classification  
num_epochs = 1000  
learning_rate = 0.001  
batch_size = 64  
# Sample data (replace with your own data)  
X = torch.randn(1000, input_size)
y = torch.randint(0, 2, (1000, 1)).float() #Binary labels (0 or 1)  
#---------------------------------
# with cpu
print('------------------------------------------')
print('======================Train with cpu======================')
start = time.time()
#Create model, optimizer, and loss function  
model = SimpleDNN(input_size, hidden_size, num_classes)  
criterion = nn.BCELoss() # Binary Cross Entropy Loss  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  

# Training loop  
for epoch in range(num_epochs):  
    # Forward pass  
    outputs = model(X)  
    loss = criterion(outputs, y)  

    # Backward and optimize  
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  

    if (epoch+1) % 10 == 0:  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')  


print("Training finished.")  

end = time.time()
print("Run time [s]: ",end-start)
#--------------------------------------------------
print('------------------------------------------')
print('======================Train with send to cuda======================')

# send to Cuda
start = time.time()
# Device configuration  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

# Create model, optimizer, and loss function  
model = SimpleDNN(input_size, hidden_size, num_classes).to(device)  
criterion = nn.BCELoss() # Binary Cross Entropy Loss  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  

# Sample data (replace with your own data)  
X =X.to(device)  
y = y.to(device) #Binary labels (0 or 1)  


# Training loop  
for epoch in range(num_epochs):  
    # Forward pass  
    outputs = model(X)  
    loss = criterion(outputs, y)  

    # Backward and optimize  
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  

    if (epoch+1) % 10 == 0:  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')  


print("Training finished.")  

end = time.time()
print("Run time [s]: ",end-start)



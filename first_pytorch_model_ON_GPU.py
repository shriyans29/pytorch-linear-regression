import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

#---------------------------------SETTING UP DEVICE AGNOSTIC CODE-------------------------------
"""
this code just tells pytorch to use gpu when available and cpu when its not
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")

#--------------------------------DATA----------------------------------
"""
creating data using linear regression model
"""
#creating parameters and data
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start,end,step).unsqueeze(dim=1)
Y = weight*X + bias

#splitting data
train_split = int(len(X)*0.8)
X_train = X[:train_split]
Y_train = Y[:train_split]
X_test = X[train_split:]
Y_test = Y[train_split:]

#visualising data
def plot_predictions(train_Data = X_train,
                     train_Lables = Y_train,
                     test_Data = X_test,
                     test_Lables = Y_test,
                     predictions = None):
    plt.figure(figsize=(10,7))

    plt.scatter(train_Data.cpu().numpy(), train_Lables.cpu().numpy(), c="b", s=4 , label = "training")

    plt.scatter(test_Data.cpu().numpy(), test_Lables.cpu().numpy(), c="g" , s=4 , label = "test")

    if predictions is not None :
        plt.scatter(test_Data.cpu().numpy(), predictions.detach().cpu().numpy(), c="r" , s=4 , label = "predictions")

    plt.legend(prop = {"size" : 14})
    plt.show()
# plot_predictions()

#--------------------------------------CREATING MODEL----------------------------------

class LinearRegressionModelV2(nn.Module):
    def __init__(self) :
        super().__init__()
        #defining parameters usning nn.linear() rather than manually
        self.linear_layer = nn.Linear(in_features=1,#in feature is basically what we input
                                                    #in features is equal to the dimention of ur input tensor aka X in this case

                                      out_features=1)#out feature is what we want it to predict aka y
                                                     #out features is equal to the number of things we want the model to classify the data into
                                                     
    def forward(self,x: torch.Tensor) -> torch.Tensor: #the torch.Tensors just says to take x input as tensor and give output as tensor
            return self.linear_layer(x)
    
#set manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print(model_1,model_1.state_dict())

#checking what device were currently using to run model
print(next(model_1.parameters()).device)

#sending model to gpu 

model_1.to(device)
print(next(model_1.parameters()).device)

#-----------------------------------------TRAINING AND TESTING---------------------------------------

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.01)

epochs = 200

#PUTTING DATA ON SAME DEVICE
X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)

for epoch in range(epochs):
     model_1.train()

     #forward pass
     y_pred = model_1(X_train)

     #calculate loss
     loss = loss_fn(y_pred,Y_train)

     #optimizer zero grad
     optimizer.zero_grad()

     #back propagation
     loss.backward()

     #optimizer step
     optimizer.step()

     #TESTING
     model_1.eval()
     with torch.inference_mode():
          test_pered = model_1(X_test)

          test_loss = loss_fn(test_pered,Y_test)

          if epoch % 10 == 0:
               print(f"epoch {epoch} | loss {loss} | test loss {test_loss}")
# print(model_1.state_dict())

#visualising predictions
plot_predictions(predictions=test_pered)

#--------------------------------SAVING AND LOADING-------------------------------
#saving
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)

MODEL_NAME = "01_first_pytoch_model_ON_GPU.pth"
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

print(MODEL_SAVE_PATH)
torch.save(obj=model_1.state_dict(),
           f=MODEL_SAVE_PATH)

#loading
loaded_model_1 = LinearRegressionModelV2()

loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

loaded_model_1.to(device)

print(next(loaded_model_1.parameters()).device, loaded_model_1.state_dict())

#evaluating loaded model
loaded_model_1.eval()
with torch.inference_mode():
     loaded_model_1_preds = loaded_model_1(X_test)
print(test_pered == loaded_model_1_preds)
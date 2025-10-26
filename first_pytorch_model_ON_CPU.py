import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# print(torch.__version__)

#------------------------------CREATING AND PREPPING DATA--------------------------------

"""using linear regression formula:
Y = A + Bx
we will make a meodel with known parameters
"""

#CREATING PARAMETERS
weight = 0.7 #  B
bias = 0.3   #  A

start = 0
end = 1
step = 0.02
X = torch.arange(start,end,step).unsqueeze(dim=1) # we use captal x as capitals are used to represent matrices
# unsqueese is needed as pytoch expects 2D input as in pytorch,
#EACH ROW = one training example
#EACH COLUMN = one feature
#aka we need to add a empty dimention for each feature
#feature is number of types of input that we will give the model while testing and trining

Y = weight * X + bias
# print(X[:10],Y[:10]) # the ':10' reffers to  everything from the starting index to the one before 10th index 
# # if we do '10:' it reffers to stuff from 10th index to the last index
# print(len(X),len(Y))

#SPLITTING DATA INTO TRAINING AND TESTING
"""
usually it is taken as 60-80% as training data
then 10-20% is taken as validation set data
and the rest 10-20% is taken as test set data
validation set is usually not used
"""
train_split = int(0.8*len(X))
X_train = X[:train_split]
Y_train = Y[:train_split]
X_test = X[train_split:]
Y_test = Y[train_split:]

# print(len(X_train),len(Y_train),len(X_test),len(Y_test))

#VISUALISING DATA

def plot_predictions(TrainData = X_train,
                     TrainLables = Y_train,
                     TestData = X_test,
                     TestLables = Y_test,
                     predictions = None):
    plt.figure(figsize=(10,7))

    plt.scatter(TrainData, TrainLables , c="b", s=4 , label = "training data")

    plt.scatter(TestData,TestLables,c="g",s=4,label = "test data")

    if predictions is not None:
        plt.scatter(TestData,predictions,c="r",s=4,label = "predictions")

    plt.legend(prop={"size":14})
    print("function ran")
    plt.show()

# plot_predictions()

#------------------------------CREATING THE MODEL-----------------------
"""
what our model does:
it see the data and tries to get a number as close to the set weight and bias,
aka it just tries to find the weight and bias parameter values

it does this through 2 main stuff:
1. gradient decent (basically it tries to find the lowest pointin the graph of loss/error that occurs between predictions it does this by taking small steps and seeing if the slope went down or up and changes the bias and weights according to that using backpropagation)
2. back propagation (this is just returning back to tune the random biases and weights aka like coming back to repair something after checking if it works or not)

gradient decent and back propagation have been implmented by pytorch itself so we dont need to create it from scratch
"""
#everything below is to do the samething we did while creating the parameters just now we are starting with random numbers rather than defining the parameters

class LinearRegressionModel(nn.Module): # almost everything in pytorch is made using a nn module this is just to tell pytorch that this is a model that will update
    def __init__(self):
        super().__init__()
        #initialise parameters
        self.weights = nn.Parameter(torch.randn(1,  # giving random start value
                                               requires_grad=True,  #telling that it can change the value using gradient decent
                                               dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,     # giving random start value
                                             requires_grad=True,    #telling that it can change the value using gradient decent
                                             dtype=torch.float))
        
        #this is the computational part that happens at each itteration (can think of it like a loop)
    def forward(self, X:torch.tensor)-> torch.tensor: # type: ignore #X is the input
            return self.weights*X +self.bias #this is the linear regression prt
        
"""
troch.nn- contains all the building blocks of computational graphs(aka neural network)
torch.nn.parameter- defines the parameters that we want our model to try to learn
torch.nn.module- its like the base class for all neural network modules as this tells the pytorch to update the parameters n stuff in this while u subclass it u need to override forward()
torch.optim- all the optimisers are in  this
def forward()- this the the computational part that happens at each itteration and all nn.module dubclasses need their own forward function
"""
#CHECKING CONTENTS OF OUR MODEL
torch.manual_seed(42)

model_0 = LinearRegressionModel()
# print(list(model_0.parameters())) 
# print(model_0.state_dict())
#both the above ones do the same thing the only difference is that state_dict give the names of the parameters too 

#-----------------------MAKING PREDICTIONS-------------------------

#we will be using torch.inference_mode()

with torch.inference_mode():
    Y_preds = model_0(X_test)
# print(Y_preds)

# # with torch.no_grad  #no grad basically stands for no gradient aka it does the same stuff as inference mode
# with torch.no_grad():
#     Y_preds =model_0(X_test)
# print(Y_preds)

# plot_predictions(predictions=Y_preds)

#-------------------------------TRAINING THE MODEL----------------------
#LOSS FUNCTIONS
"""
the idea of training is basically helping the model to guess the right parameter,
or to know how bad a model is representing the data it has to predict(to visualise- how far the red dots are from the green,
aka rn the red dots are a bad rep of the green dots as its just using random numbers)
this prediction of loss is done by something called loss function

what all we need to train:-

1.) Loss Functions:- this is basically a function to measure how wrong ur model is compared to the idea outputs aka lower is better
2.) Optimizer:- adjusts the parameters of the model according to the loss to reduce the loss function

there are many diff optimisers and loss functions are which to use comes through experiance
"""
# setting up the loss function
loss_fn = nn.L1Loss()

#setting up the optimiser
optimiser = torch.optim.SGD(params=model_0.parameters(), # this part is just telling the optimiser what are the parameters it has to change
                            lr=0.01) # lr = learning rate    basically what this does is tells optimiser how much to change the value of the parameters,
                                     # aka if the parameter is 0.3364 and lr is 0.1 itll affect the first 3 if it is 0.001 itll affect the 6,
                                     # aka its telling how much to change the parameters in each itteration
                                     # it is also called a hyperparameter these are just parameters that we devs set

# BUILDING TRAINING LOOPS
"""
STUFF GOING ON IN TRAINING LOOPS
0) loop through data
1) Forward pass (sending the data through the 'forward()' function) to make predictions on data - also called forward propagation
2) calculate the loss (compare forward pass predictions to lables)
3) optimiser zero grad
4) Loss backward - move backwards through the network to calculate the gradients of each of the parameters of our model with respect to the loss (back propagation)
5) optimizer step - use the optimizer to adjust parameters and reduce loss (gradient decent)

optimizer.zero_grad() :-
loss.backward() → writes new ink.

optimizer.step() → uses the ink to move weights.

optimizer.zero_grad() → erases the board, so the next pass starts clean.
"""

epochs = 300 #epochs means 1 itteration through the data this is also a hyper parameter

# tracking different values
epoch_count = []
loss_values = []
test_loss_values = []
#STEP 0 LOOP THROUGH
for epoch in range(epochs):
    #set model to training mode
    model_0.train() #this tells pytorch that the model is to be trained not tested and u can do stuff like randomly set some nodes activation to 0 or stuff like that

    #Forward pass
    Y_preds=model_0(X_train)

    #calculating loss
    loss = loss_fn(Y_preds, Y_train)
    # optimizer zero grad
    optimiser.zero_grad() #we do this so that after each epoch the value of the gradient does not overlap and the value og the gradient is 0
    # optimizer.zero_grad() → erases the board, so the next pass starts clean.

    # back propagation
    loss.backward() # back propagation


    #step the optimizer ( perform gradient decent)
    optimiser.step() #changes the weights according to loss
                     # this basically checks the slope and decides if we need to  take a big step or a small step in  the gradient graph to reach the min point

    #set model to evaluation/test
    model_0.eval() #this tlls pytorch that the model is to be tested now and to shut off all the random shenanigens like setting nodes to not activate n stufff
    with torch.inference_mode():
        # Forward pass
        test_pred = model_0(X_test)

        # calculate loss
        test_loss = loss_fn(test_pred , Y_test)
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            loss_values.append(loss.item())
            test_loss_values.append(test_loss.item())
        #     print(f"epoch: {epoch} | loss: {loss}| test loss {test_loss}")
        #     print(model_0.state_dict())


# TO GET BETTER AND MORE ACCURATE RESULTS WE ALTER THE LR AND EPOCH TOGETHER NOT THE AMOUNT OF DATA WE GIVE IT 
# ONCE THE LR STARTS TO REACH ITS LIMIT THAT IS BY REACHING A CERTAIN POINT WHERE,
# THE OTHER SIDE OF THE DECENT GRAPH IS SO LOW THAT THE STEPS JUST KEEP JUMPING FROM ONE SIDE TO THE OTHER
# HENCE TO IMPROVE THIS WE NEED BEIIGER STEPS AT THE START TO CLOSE BIGGER GAP IN THE START AND REDUCE THE LR IN SOME INTERVALS TO GET SMALLER STEPS AS WE GET TO THE REQUIRED VALUES

# print(f"loss: {loss}")
# print(model_0.state_dict())

with torch.inference_mode():
     Y_preds_50_epoch = model_0(X_test)

plot_predictions(predictions=Y_preds_50_epoch)

# PLOTTING TRAINING AND TEST LOSS CURVES
plt.plot(epoch_count,test_loss_values,label = " test loss")
plt.plot(epoch_count,loss_values,label = "train loss")
plt.title("training and test loss curves")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend()
plt.show()

#----------------------SAVING AND LOADING A MODEL---------------------------

"""
we save and load the model so that we dont have to train it everytime as vs code and everything saves the generated weights and biases in the ram hence once we close the app the ram is 

torch.save() - allows you to save pytoch object in pythons pickle fromat
troch.load() - aloows you to load a saved pytorch model
torch.nn.module.load_state_dict() - allows u to load a models saved state dictionary
"""

#SAVING
"""
mostly we only save the state dict as to saving the whole model because 
1. saving the state dict creates a smaller file than saving the whole model
2. saving the state dict only save the parameters so we can load into a diff model with with diff variable names an stuff and itll still work
"""
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)

MODEL_NAME = "01_first_pytorch_model_ON_CPU.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(MODEL_SAVE_PATH)

torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)

print(os.listdir("models"))

#LOADING
"""
as we saved the models state_dict() rather than the entire model 
we can create a new instance of our model class and load the saved state_dict() into that
"""

#initiating new instance of our model

loaded_model_0 = LinearRegressionModel()

# print(loaded_model_0.state_dict())
#laoading state dict into new instance

loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))# f refefers to the file path
print(loaded_model_0.state_dict())

#making predictions with loaded model

loaded_model_0.eval()
with torch.inference_mode():
     loaded_model_preds = loaded_model_0(X_test)
print(loaded_model_preds)

print(loaded_model_preds == test_pred)

print(next(model_0.parameters()).device)
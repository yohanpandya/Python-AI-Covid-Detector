#!/usr/bin/env python
# coding: utf-8

# In[1]:


#q1:
#about how many bytes does trainX consume?
import torch
import matplotlib.pyplot as plt
import pandas as pd
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

trainX = torch.tensor(train_df.iloc[:, :-1].values, dtype=torch.float64)
trainY = torch.tensor(train_df.iloc[:, -1:].values, dtype=torch.float64)

testX = torch.tensor(test_df.iloc[:, :-1].values, dtype=torch.float64)
testY = torch.tensor(test_df.iloc[:, -1:].values, dtype=torch.float64)
#print (testY)
#print (trainY)

totalbytes = 0;
for i in trainX:
    for j in i:
        totalbytes= totalbytes + trainX.element_size() 
print("CELL 1: total bytes in trainX:", totalbytes)
    



# In[2]:


#q2: what is the biggest difference we would have any one cell if we used float16 instead of float64?

trainX_float16 = trainX.to(dtype=torch.float16)

difference = torch.abs(trainX - trainX_float16)

max_difference = difference.max().item()

print("CELL 2: biggest different in any one cell if we used float16 instead of float64",max_difference)


# In[3]:


#q3: Write a code snippet to produce a True/False answer.
cuda_available = torch.cuda.is_available()

if(cuda_available):
    print("CELL 3: cuda GPU is available")
else:
    print("CELL 3: cuda GPU is not available")


# In[4]:


#q4: what is the predicted number of deaths for the first census tract?
#PART 2
coef = torch.tensor([
        [0.0040],
        [0.0040],
        [0.0040],
        [0.0040],
        [0.0040],
        [0.0040], # POS_50_59_CP
        [0.0300], # POS_60_69_CP
        [0.0300],
        [0.0300],
        [0.0300]
], dtype=trainX.dtype)
coef
first_row = testX[0]
#testX[0]
#firstRowResult  = torch.matmul(first_row, coef).item()
firstRowResult = (first_row@coef).item()
print("CELL 4: using fixed coefs, the number of predicted deaths for the first census tract is",firstRowResult)


# In[5]:


#q5: what is the average number of predicted deaths, over the whole testX dataset?
TrainXResult = torch.matmul(testX,coef)
mean = torch.mean(TrainXResult)
print("CELL 5: the average number of predicted deaths over the whole testX dataset is",mean.item())


# In[6]:


#q6: first, what is y when x is a tensor containing 0.0?
#PART 3: Optimization
def f(x):
    return x*x - 8*x + 19
#x = torch.arange(-8,5,0.1)
#y=f(x)

x = torch.tensor(0.0, requires_grad = True)
#plt.plot(x,y)
y = f(x)
plt.plot(x.detach(),y.detach(),"ro",markersize = 3)
print("CELL 6: what is the prediction when x (optimization variable), has just been optimized?" ,y.item())


# In[7]:


#q7: what x value minimizes y?
def f(x):
    return x**2 - 8*x + 19
#x = torch.arange(-8,5,0.1)
#y=f(x)
#plt.plot(x,y)


#x = torch.tensor(0.0,requires_grad = True)
optimizer = torch.optim.SGD([x],lr = 0.1)
for epoch in range(100):
    y = f(x)
    
    #plt.plot(x.detach(),y.detach(),"ro",markersize = 3+epoch*0.1)
    y.backward()
    optimizer.step()
    optimizer.zero_grad()
x.item()
print("CELL 7: after using SGC optimizer with LR = 0.1, the x value that minimizes y is", x.item())

# In[ ]:





# In[8]:


#q8 
#what is the MSE (mean-square error) when we make predictions using this vector of zero coefficients?
loss_fn = torch.nn.MSELoss()
coef = torch.zeros((10,1), dtype = torch.float64, requires_grad = True)
#trainX.shape
#trainY.shape
optimizer = torch.optim.SGD([coef], lr = 0.1)



predictions = trainX @ coef
loss = loss_fn(predictions, trainY)


loss.backward() #computes coef.grad
optimizer.step()
loss.item()
print("CELL 8: MSE when we make prediction of vector of zero coefs (in order to see the difference when we optimize x", loss.item())





# In[9]:


coef


# In[10]:


trainX.shape, coef.shape


# In[11]:


#q9
torch.manual_seed(544)
coef = torch.zeros((10,1), dtype = trainX.dtype, requires_grad = True)
ds = torch.utils.data.TensorDataset(trainX, trainY)
dl = torch.utils.data.DataLoader(ds, batch_size=50, shuffle=True)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD([coef], lr = 0.000002)

for epoch in range(500):
    for batch_x, batch_y in dl:
        #print(len(batch_x))
        #print(len(batch_y))
        predictions = batch_x @ coef
        loss = loss_fn(predictions, batch_y)
        loss.backward()
        optimizer.step()  # Update coefficients
        optimizer.zero_grad()


print("CELL 9: used batch gradient descent to update coefs")


# In[12]:


#q10: 
# torch.manual_seed(544)
# coef = torch.zeros((10,1), dtype = testX.dtype, requires_grad = True)
# test
# ds = torch.utils.data.TensorDataset(testX, testY)
# dl = torch.utils.data.DataLoader(ds, batch_size=50, shuffle=True)
# loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.SGD([coef], lr = 0.000002)

# for epoch in range(500):
#     for batch_x, batch_y in dl:
#         #print(len(batch_x))
#         #print(len(batch_y))
#         batch_y.
#         predictions = batch_x @ coef
#         loss = loss_fn(predictions, batch_y)
#         loss.backward()
#         optimizer.step()  # Update coefficients
#         optimizer.zero_grad()


print ("CELL 10: actual loss",loss_fn(testX @ coef,testY).item())

import numpy as np 
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from scipy.stats import truncnorm
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from plotnine import ggplot, geom_point,geom_line, aes, stat_smooth, facet_wrap


base = importr('base')
stats = importr('stats')
ro.r['source']('pHAbs_NLSNLMM.R')

##

data = pd.read_csv("Full_pHAbsdata.csv")
data.sample(frac=1)

with localconverter(ro.default_converter + pandas2ri.converter):
  NLSresult = ro.r.Fit_NLS(data)
  NLMMresult = ro.r.Fit_NLMM(data)


data["Ahat_NLS"] = np.array(stats.predict(NLSresult))
data["Ahat_NLMM"] = np.array(stats.predict(NLMMresult))

(ggplot(data,aes('pH','ALED',color ='factor(Trial)')) 
+ geom_point() + geom_line(aes('pH','Ahat_NLMM',color='factor(Trial)'))
+ geom_line(aes('pH','Ahat_NLS'),inherit_aes=False))

def generate_pHAbs(n,Amax=0.43,pKa=7.47,phi=0.46,sd_e=0.025):
    mean_pH,sd_pH = 7.6, 2.2
    min_pH, max_pH = 0, 14
    a,b = (min_pH - mean_pH)/sd_pH , (max_pH-mean_pH)/sd_pH
    pH = truncnorm.rvs(a,b,loc=mean_pH,scale=sd_pH,size=n)
    e = np.random.normal(loc=0,scale=sd_e,size=n)
    A = Amax / (1+(np.exp(pKa-pH))/phi) + e
    simdf = pd.DataFrame({'pH': pH,'ALED': A})
    return simdf

def generate_pHAbs_Trials(Trials,n,Amax=0.43,Asd=0.04,pKa=7.47,phi=0.46,sd_e=0.025):
    Amaxes = np.random.normal(Amax,Asd,Trials)
    simdfall = []
    for i in range(Trials):
        simdf = generate_pHAbs(n=n,Amax=Amaxes[i],pKa=pKa,phi=phi,sd_e=sd_e)
        simdf['Trial'] = i+1 
        simdfall.append(simdf)
    simdfall = pd.concat(simdfall)
    return simdfall 


class pHAbsDataset(Dataset):
    def __init__(self,pH,Abs):
        self.pH=pH.reshape(-1,1)
        self.Abs = Abs.reshape(-1,1)
    
    def __len__(self):
        return len(self.pH)
    
    def __getitem__(self,idx):
        return self.pH[idx],self.Abs[idx]



class pHAbsLayer(nn.Module):
    """Custom pHAbs Layer: Amax/(1+e^(pKa-pH)/phi)"""
    def __init__(self):
        super().__init__()
        weights = np.random.normal([1,7.6,0.5],[0.2,0.5,0.1]) #[Amax,pKa,phi]
        weights = torch.from_numpy(weights)
        self.weights = nn.Parameter(weights)
        self.regularizer = torch.zeros(3,dtype=torch.float64)

    def forward(self,x):
        y = self.weights[0]/(1+torch.exp((self.weights[1]-x)/self.weights[2]))
        return y 


class pHAbsModel(nn.Module):
    def __init__(self,lam_Amax=0,lam_pKa=0,lam_phi=0):
        super().__init__()
        self.f_pH = pHAbsLayer()
        self.f_pH.regularizer[0] = lam_Amax
        self.f_pH.regularizer[1] = lam_pKa
        self.f_pH.regularizer[2] = lam_phi 

    def forward(self,x):
        return self.f_pH(x)


def penalty(model):
    weights = model.f_pH.weights
    regularizer = model.f_pH.regularizer
    prior = torch.Tensor([0,7.6,1/np.log(10)]) 
    penalty = (weights-prior).abs().dot(regularizer)
    return penalty 


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        pen = penalty(model)

        pen_loss = loss + pen
        # Backpropagation
        optimizer.zero_grad()
        pen_loss.backward() 
        #loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        return(loss)


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            

    #test_loss /= size
    print(f"Avg loss: {test_loss:>8f} \n")
    return(test_loss)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(100)
TrainSim = generate_pHAbs_Trials(Trials=100,n=100)
np.random.seed(10)
ValSim = generate_pHAbs_Trials(Trials=100,n=100,Amax=0.40,pKa=7.52,phi=0.48)

pH_Train, Abs_Train = TrainSim.pH.to_numpy(), TrainSim.ALED.to_numpy()
pH_Val,Abs_Val = ValSim.pH.to_numpy(), ValSim.ALED.to_numpy()

TrainDS = pHAbsDataset(pH_Train,Abs_Train)
ValDS = pHAbsDataset(pH_Val,Abs_Val)

TrainLoader = DataLoader(TrainDS,batch_size=100,shuffle=True)
ValLoader = DataLoader(ValDS,batch_size=100,shuffle=True)

sim_model = pHAbsModel()
learning_rate = 0.01

loss_fn_train = nn.MSELoss()
loss_fn_val = nn.MSELoss(reduction="sum") #because test loop divides in the end

optimizer = torch.optim.Adam(sim_model.parameters(), lr=learning_rate)

epochs = 1000
loss_simtrain = np.zeros(epochs)
loss_simval = np.zeros(epochs)

for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------------")
    loss_simtrain[i] = train_loop(TrainLoader, sim_model, loss_fn_train, optimizer)
    loss_simval[i] = test_loop(ValLoader,sim_model,loss_fn_val)

sim_model.f_pH.weights 

def pHAbsfun(theta,pH,Aobs):
    A = theta[0]/(1+np.exp((theta[1]-pH)/(theta[2])))
    res = A-Aobs
    return res

pHAbstrain_fun = lambda theta: pHAbsfun(theta,pH_Train,Abs_Train)

ls_result = least_squares(pHAbstrain_fun,[0.5,7.6,0.4])


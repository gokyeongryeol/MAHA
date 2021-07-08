import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from data_loader import *
from MAHA_regression import *
import numpy as np

import torch
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import time
import random

matplotlib.use('Agg')

def main(data, shot, phase, n_epoch, KD_w, index=None):
    loader = Heterogeneous_dataset(batch_size=25, update_batch_size=shot, update_batch_size_eval=10, num_classes=1, index=index, data_source='1D', is_train=True)
    
    model = MAHA(dim_input=1, dim_output=1, ln=True, n_way=1, phase=phase, batch_size=25, num_inds=32, dp_rate=0.0).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    for epoch in range(n_epoch):
        model.train()
        
        init_inputs, outputs, _, _ = loader.generate_batch(is_test=False)    
            
        Cx = torch.tensor(init_inputs[:, :1*shot, :], dtype=torch.float)
        Cy = torch.tensor(outputs[:, :1*shot, :], dtype=torch.float)
        Tx = torch.tensor(init_inputs[:, :, :], dtype=torch.float)
        Ty = torch.tensor(outputs[:, :, :], dtype=torch.float)
        
        _, train_KL, train_NLL, train_KD, train_MSE = model(Cx.to(device), Cy.to(device), Tx.to(device), Ty.to(device)) 
        train_loss = train_KL + train_NLL + KD_w * train_KD
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
                    
        with torch.no_grad():
            model.eval()
        
            init_inputs, outputs, _, _ = loader.generate_batch(is_test=True)    
        
            Cx = torch.tensor(init_inputs[:, :1*shot, :], dtype=torch.float)
            Cy = torch.tensor(outputs[:, :1*shot, :], dtype=torch.float)
            Tx = torch.tensor(init_inputs[:, :, :], dtype=torch.float)
            Ty = torch.tensor(outputs[:, :, :], dtype=torch.float)
        
            _, test_NLL, test_MSE = model(Cx.to(device), Cy.to(device), Tx.to(device), Ty.to(device))
            test_loss = test_NLL
                    
        if (epoch+1) % 100000 == 0:
            print('[Epoch %d] KL: %.3f, NLL: %.3f, train_loss: %.3f, test_loss: %.3f, train_MSE: %.3f, test_MSE: %.3f' 
                  % (epoch+1, train_KL, train_NLL, train_loss, test_loss, train_MSE, test_MSE))
            torch.save(model.state_dict(), '../models/'+data+str(shot)+phase+'/'+str(epoch+1)+'.pt')

if __name__ == '__main__':  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data = 'mixture'
    shot = 5
    phase = 'finetuning' # or 'pretrain'
    n_epoch = 1000000
    
    tic = time.time()
    
    #pretrain
    #main(data, shot, phase, n_epoch, 0.0, index=None)
    
    #finetuning, dataloader0
    main(data, shot, phase, n_epoch, 0.12086667050439993, index=0)
    
    #finetuning, dataloader1
    #main(data, shot, phase, n_epoch, 0.9413495232210947, index=1)
    
    toc = time.time()
    
    mon, sec = divmod(toc-tic, 60)
    hr, mon = divmod(mon, 60)
    print('total wall-clock time is ', int(hr),':',int(mon),':',int(sec))
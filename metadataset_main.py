import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from data_loader import *
from MAHA_variable import *
from utils import *
import pickle
import numpy as np

import torch
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import time
import random


matplotlib.use('Agg')

def main(data, phase, n_epoch, KL_w, lr):
    train_loader = Meta_dataset(batch_size=16, is_train=True, is_test=False)
    valid_loader = Meta_dataset(batch_size=16, is_train=True, is_test=True)
    
    model = MAHA(dim_input=84*84*3, ln=True, phase=phase, batch_size=16, num_inds=256).to(device)
    
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
        
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    KL_list, NLL_list = [], []
    train_loss_list, test_loss_list = [], []
    train_accur_list, test_accur_list = [], []
     
    for epoch in range(n_epoch):
        model.train()
        
        Cx, Cy, Tx, Ty, _ = train_loader.generate_batch()
        Cx = [x.to(device) for x in Cx]
        Cy = [y.to(device) for y in Cy]
        Tx = [x.to(device) for x in Tx]
        Ty = [y.to(device) for y in Ty]
        
        train_KL, train_NLL, train_accur = model(Cx, Cy, Tx, Ty)
        train_loss = KL_w * train_KL + train_NLL
        
        KL_list.append(train_KL.item())
        NLL_list.append(train_NLL.item())
        train_accur_list.append(train_accur.item())
        train_loss_list.append(train_loss.item())
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
             
        with torch.no_grad():
            model.eval()
        
            Cx, Cy, Tx, Ty, sel_set = valid_loader.generate_batch()
            Cx = [x.to(device) for x in Cx]
            Cy = [y.to(device) for y in Cy]
            Tx = [x.to(device) for x in Tx]
            Ty = [y.to(device) for y in Ty]
            
            test_NLL, test_accur = model(Cx, Cy, Tx, Ty)
            test_loss = test_NLL
            test_accur_list.append(test_accur.item())
            test_loss_list.append(test_loss.item())
    
        if (epoch+1) % 1000 == 0:
            print('[Epoch %d] train_KL: %.3f, train_NLL: %.3f, train_loss: %.3f'
                  % (epoch+1, train_KL, train_NLL, train_loss))
            
            torch.save(model.state_dict(), '../models/'+data+phase+'/'+str(epoch+1)+'.pt')
            
            torch.save(KL_list, '../loss/'+data+phase+'/KL_list.pt')
            torch.save(NLL_list, '../loss/'+data+phase+'/NLL_list.pt')
            torch.save(train_loss_list, '../loss/'+data+phase+'/train_loss_list.pt')
            torch.save(test_loss_list, '../loss/'+data+phase+'/test_loss_list.pt')
            torch.save(train_accur_list, '../loss/'+data+phase+'/train_accur_list.pt')
            torch.save(test_accur_list, '../loss/'+data+phase+'/test_accur_list.pt')
        
        
if __name__ == '__main__':  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    data = 'metadataset'
    phase = 'pretrain'
    n_epoch = 100000

    tic = time.time()
    #hyperparameter search
    for itr in range(0, 10):
        KL_w = np.random.uniform(1e-4, 1e-1)
        lr = np.random.uniform(1e-6, 1e-4)
        print(KL_w, lr)
        with open(data+str(itr)+'th'+phase, 'w') as f:
            f.write(str(KL_w) + '/' + str(lr) + '\n')
        data_prime = data+str(itr)+'th'
        main(data_prime, phase, 5000, KL_w, lr)
    
    toc = time.time()
    
    mon, sec = divmod(toc-tic, 60)
    hr, mon = divmod(mon, 60)
    print('total wall-clock time is ', int(hr),':',int(mon),':',int(sec))
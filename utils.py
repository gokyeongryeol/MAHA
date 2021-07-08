import os
import torch
import torch.nn.functional as F

import imageio
import math
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pandas as pd

from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_regression(model, mixture, num_way, num_shot, result_path):   
    init_inputs = torch.load('eval/'+mixture+'_inp_'+str(num_shot)+'_shot.pt')
    outputs = torch.load('eval/'+mixture+'_out_'+str(num_shot)+'_shot.pt')
    
    #interpolation
    rng = np.random.default_rng(seed=2)
    random_idx = rng.permutation(init_inputs.shape[1])
    
    #extrapolation
    tmp = int(init_inputs.shape[1]*0.1)
    random_idx = np.concatenate((rng.permutation(tmp), np.arange(init_inputs.shape[1])[tmp:]))  
    
    Cx = torch.tensor(init_inputs[:, random_idx[:num_way*num_shot], :], dtype=torch.float)
    Cy = torch.tensor(outputs[:, random_idx[:num_way*num_shot], :], dtype=torch.float)
    
    Tx = torch.tensor(init_inputs[:, :, :], dtype=torch.float)
    Ty = torch.tensor(outputs[:, :, :], dtype=torch.float)
    
    y_dist, _, _, MSE = model(Cx.to(device), Cy.to(device), Tx.to(device), Ty.to(device))
           
    Cx, Cy = torch.squeeze(Cx, -1).cpu(), torch.squeeze(Cy, -1).cpu()
    Tx, Ty = torch.squeeze(Tx, -1).cpu(), torch.squeeze(Ty, -1).cpu()
    
    mean, std = y_dist.mean.detach(), y_dist.stddev.detach()
    mean, std = torch.squeeze(mean, -1).cpu(), torch.squeeze(std, -1).cpu()
    
    plt.figure()
    plt.title('regression')
    plt.plot(Tx[0], Ty[0], 'k:', label='True')
    plt.plot(Cx[0], Cy[0], 'b^', markersize=10, label='Contexts')
    plt.plot(Tx[0], mean[0], 'g-', label='Predictions')
    plt.fill(torch.cat((Tx[0], torch.flip(Tx[0], [0])),0),
             torch.cat((mean[0] - 1.96 * std[0], torch.flip(mean[0] + 1.96 * std[0], [0])),0),
             alpha=.5, fc='g', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(min(Ty[0]) - 0.3 * (max(Ty[0]) - min(Ty[0])), max(Ty[0]) + 0.3 * (max(Ty[0]) - min(Ty[0])))
    plt.savefig(result_path+mixture+'.png')
    plt.close()


def plot_function_scatter(model, test_loader, num_way, num_shot, data_source, result_path):
    color_list = ['red', 'green', 'blue', 'black']
    color_list2 = ['orange', 'purple']
    label_list = ['Sine', 'Line', 'Quadratic', 'Cubic']
    
    kind_total = torch.tensor([])
    emb_total = torch.tensor([])
    for itr in range(100):
        init_inputs, outputs, _, sel_set = test_loader.generate_mixture_batch(is_test=True)

        rng = np.random.default_rng()
        random_idx = rng.permutation(init_inputs.shape[1])

        Cx = torch.tensor(init_inputs[:, random_idx[:num_way*num_shot], :], dtype=torch.float).to(device)
        Cy = torch.tensor(outputs[:, random_idx[:num_way*num_shot], :], dtype=torch.float).to(device)
        
        task_param = model.param_encoder(model.x_emb(Cx), Cy, 'global').mean(1, True)
        emb = task_param[:, :, :int(task_param.size(-1)/2)]
        emb = emb.cpu().detach()
        
        kind_total = torch.cat((kind_total, torch.tensor(sel_set, dtype=torch.float)), 0)
        emb_total = torch.cat((emb_total, emb), 0)
    
    X_embedded = TSNE(n_components=2).fit_transform(emb_total.view(-1, 256))
    
    for class_idx in range(0, 4):
        index_list = kind_total == class_idx
        plt.scatter(X_embedded[index_list,0], X_embedded[index_list,1], c=color_list[class_idx], marker='.', label=label_list[class_idx]) 
    plt.savefig(result_path+'_function_scatter.png')
    plt.close()

    mergings = linkage(X_embedded, method='average')
    prediction = fcluster(mergings, 2, criterion='maxclust')
    predict = pd.DataFrame(prediction)
    predict.columns=['predict']
    labels = pd.DataFrame(kind_total)
    labels.columns=['labels']

    ct = pd.crosstab(predict['predict'],labels['labels'])
    print(ct)
    
    for class_idx in range(2):
        index_list = prediction == class_idx + 1
        plt.scatter(X_embedded[index_list,0], X_embedded[index_list,1], c=color_list2[class_idx], marker='.', label=label_list[class_idx]) 
    plt.savefig(result_path+'_agglomerative_scatter.png')
    plt.close()

    
def plot_image_scatter(model, test_loader, num_way, num_shot, data_source, result_path):
    color_list = ['red', 'green', 'blue', 'black']
    color_list2 = ['brown', 'purple', 'gray', 'orange']
    
    label_list = ['Bird', 'Texture', 'Aircraft', 'Fungi']
    
    kind_total = torch.tensor([])
    emb_total = torch.tensor([])
    for itr in range(1000):
        image_batch, label_batch, sel_set = test_loader.generate_multidataset_batch(is_test=True)

        Cx = torch.tensor(image_batch[:, :num_way * num_shot, :], dtype=torch.float).to(device)
        Cy = torch.tensor(label_batch[:, :num_way * num_shot, :], dtype=torch.float).to(device)
        
        task_param = model.param_encoder(model.x_emb(Cx), Cy, 'stochastic').mean(1, True)
        emb = model.stochastic_distribution(task_param).mean 
        emb = task_param[:, :, :int(task_param.size(-1)/2)]

        emb = emb.cpu().detach()
        
        kind_total = torch.cat((kind_total, torch.tensor(sel_set, dtype=torch.float)), 0)
        emb_total = torch.cat((emb_total, emb), 0)
    
    X_embedded = TSNE(n_components=2).fit_transform(emb_total.view(-1, 128))
    
    for class_idx in range(0, 4):
        index_list = kind_total == class_idx
        plt.scatter(X_embedded[index_list,0], X_embedded[index_list,1], c=color_list[class_idx], marker='.', label=label_list[class_idx]) 
    plt.savefig(result_path+'_image_scatter.png')
    plt.close()
    
    num_clusters = 4
        
    mergings = linkage(X_embedded, method='average')
    prediction = fcluster(mergings, num_cluster, criterion='maxclust')
    predict = pd.DataFrame(prediction)
    predict.columns=['predict']
    labels = pd.DataFrame(kind_total)
    labels.columns=['labels']

    ct = pd.crosstab(predict['predict'],labels['labels'])
    print(ct)
    
    for class_idx in range(num_cluster):
        index_list = prediction == class_idx + 1
        plt.scatter(X_embedded[index_list,0], X_embedded[index_list,1], c=color_list2[class_idx], marker='.', label=label_list[class_idx]) 
    plt.savefig(result_path+'_agglomerative_scatter.png')
    plt.close()

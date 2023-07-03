import torch

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from scipy import interpolate

from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer, MaxAbsScaler, StandardScaler, KBinsDiscretizer
from datasetPreProcessing import OneHotTransformer
from imblearn.over_sampling import RandomOverSampler

import time
import math
import os
import copy
import types

def loadData(dataName):
    if dataName == 'network_data':
        raw_data = np.loadtxt('../data/network_flow_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/network_flow_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
    elif dataName == 'medical_data':
        raw_data = np.loadtxt('../data/medical_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/medical_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/medical_attack_data_undetected.csv', skiprows=1, delimiter=',')[:,:-1]
    elif dataName == 'full_data':
        raw_data = np.concatenate((np.loadtxt('../data/network_flow_regular_data.csv', skiprows=1, delimiter=',')[:,:-1], np.loadtxt('../data/medical_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]), axis=1)
        anomalous_raw = np.concatenate((np.loadtxt('../data/network_flow_attack_data.csv', skiprows=1, delimiter=',')[:,:-1], np.loadtxt('../data/medical_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]), axis=1)
    elif dataName == 'network_data1':
        raw_data = np.loadtxt('../data/network_flow1_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/network_flow1_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
    elif dataName == 'medical_data1':
        #raw_data = np.loadtxt('../data/medical_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        #raw_data = np.loadtxt('../data/medical1_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        raw_data = np.concatenate((np.loadtxt('../data/medical_regular_data.csv', skiprows=1, delimiter=',')[:,:-1], np.loadtxt('../data/medical1_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]), axis=0)
        anomalous_raw = np.loadtxt('../data/medical_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
        #anomalous_raw = np.loadtxt('../data/medical_attack_data_undetected.csv', skiprows=1, delimiter=',')[:,:-1]
    elif dataName == 'full_data1':
        raw_data = np.concatenate((np.loadtxt('../data/network_flow1_regular_data.csv', skiprows=1, delimiter=',')[:,:-1], np.loadtxt('../data/medical1_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]), axis=1)
        anomalous_raw = np.concatenate((np.loadtxt('../data/network_flow1_attack_data.csv', skiprows=1, delimiter=',')[:,:-1], np.loadtxt('../data/medical1_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]), axis=1)
    elif dataName == 'network_data2':
        raw_data = np.loadtxt('../data/network_flow2_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/network_flow2_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
    return raw_data, anomalous_raw

def preProcessData_OneClass(raw_data:np.array, anomalous_data:np.array, split:tuple or list=(0.6, 0.2, 0.2), trim:bool=False, trim_threshold:float=0.96, normalize:bool=True, normalization_scheme:str='standard_scaling', cross_validation:bool=True, filterLinearDependencies:bool = True, filter_threshold:float=0.95, removeNoise:bool=False, noise_threshold:float=3, categorical_data_index=None, discretize=False, return_dict={}, args=None):

    if sum(split) != 1:
        return ValueError('\'split\' must sum to 1.')
    len_raw_data = len(raw_data)
    proportion_train, proportion_validation = split[0], split[1]
    if cross_validation:
        random_idc = np.random.choice(len_raw_data, len_raw_data, replace=False)
        raw_data = raw_data[random_idc,:]

    i, j = math.floor(len_raw_data * proportion_train), math.floor(len_raw_data * (proportion_train + proportion_validation))
    train_data, validation_data, test_data = raw_data[0:i], raw_data[i:j], raw_data[j:]
    
    # one hot encoding for categorical data
    if categorical_data_index:
        transformer = OneHotTransformer(train_data, categorical_data_index)
        train_data, validation_data, test_data, anomalous_data = transformer.transform(train_data), transformer.transform(validation_data), transformer.transform(test_data), transformer.transform(anomalous_data)
        train_data, validation_data, test_data, anomalous_data = np.float64(train_data), np.float64(validation_data), np.float64(test_data), np.float64(anomalous_data)
        print(train_data.dtype)
    # trim
    return_dict["trim_columns_to_save"] = np.array([i for i in range(train_data.shape[-1])])
    if trim:    
        columns_to_save = []
        # Trim data by filtering out useless features, which are the ones who appear (almost) the same throughout observations
        for i in range(train_data.shape[1]):
            if np.unique(train_data[:,i], return_counts=True)[1].max() < trim_threshold * train_data.shape[0]: # max frequency of occurrences
                columns_to_save.append(i)
        print("Untrimmed columns: ", columns_to_save)
        train_data, validation_data, test_data, anomalous_data = train_data[:,columns_to_save], validation_data[:,columns_to_save], test_data[:,columns_to_save], anomalous_data[:,columns_to_save] 
        # save idc
        return_dict["trim_columns_to_save"] = np.array(columns_to_save)
    print('Train data shape after trim: ', train_data.shape)
    # filter corrcoef
    return_dict["corrcoef_columns_to_save"] = return_dict["trim_columns_to_save"]
    if filterLinearDependencies:
        i, seq = 0, np.arange(train_data.shape[-1])
        while True:
            coef_matrix = abs(np.corrcoef(train_data, rowvar=False) - np.eye(train_data.shape[1])) # corr-coefficient matrix with self coefficient zeroed out
            if i >= coef_matrix.shape[0]:
                break # no more element to filter
            idc = coef_matrix[i] < filter_threshold # indices that are NOT strongly linearly dependent with i-th element
            #print("FLD" + str(i) + ": ", idc)
            train_data, validation_data, test_data, anomalous_data = train_data[:,idc], validation_data[:,idc], test_data[:,idc], anomalous_data[:,idc] 
            seq = seq[idc]
            i += 1 # go to next unfiltered feature
        # save idc
        return_dict["corrcoef_columns_to_save"] = return_dict["corrcoef_columns_to_save"][seq]
        
    print('Train data shape after filter corrcoef: ', train_data.shape)
    # discretize
    if discretize:
        transformer = KBinsDiscretizer(n_bins=[20 for _ in range(train_data.shape[-1])], encode='ordinal').fit(train_data)
        train_data, validation_data, test_data, anomalous_data = transformer.transform(train_data), transformer.transform(validation_data), transformer.transform(test_data), transformer.transform(anomalous_data)
    if normalize and not discretize:
        eps = 1e-5 # small constant to prevent divide by zero error
        print('Normalization scheme: ', normalization_scheme)
        if normalization_scheme == 'min_max_scaling':
            train_max, train_min = train_data.max(axis=0, keepdims=True), train_data.min(axis=0, keepdims=True)
            train_data = (train_data-train_min)/(train_max - train_min)
            validation_data = (validation_data-train_min)/(train_max - train_min)
            test_data = (test_data-train_min)/(train_max - train_min)
            anomalous_data = (anomalous_data-train_min)/(train_max - train_min)
        elif normalization_scheme == 'robust_scaling':
            transformer = RobustScaler(quantile_range=(20,80)).fit(train_data)
            print('transformer info: ', transformer.center_, transformer.scale_, transformer.quantile_range)
            train_data, validation_data, test_data, anomalous_data = transformer.transform(train_data), transformer.transform(validation_data), transformer.transform(test_data), transformer.transform(anomalous_data)
        elif normalization_scheme == 'quantile_transform_uniform':
            transformer = QuantileTransformer(output_distribution="uniform").fit(train_data)
            train_data, validation_data, test_data, anomalous_data = transformer.transform(train_data), transformer.transform(validation_data), transformer.transform(test_data), transformer.transform(anomalous_data)
        elif normalization_scheme == 'quantile_transform_normal':
            transformer = QuantileTransformer(output_distribution="normal").fit(train_data)
            train_data, validation_data, test_data, anomalous_data = transformer.transform(train_data), transformer.transform(validation_data), transformer.transform(test_data), transformer.transform(anomalous_data)
        elif normalization_scheme == 'power_transformation_yj':
            transformer = PowerTransformer(method="yeo-johnson").fit(train_data)
            train_data, validation_data, test_data, anomalous_data = transformer.transform(train_data), transformer.transform(validation_data), transformer.transform(test_data), transformer.transform(anomalous_data)
        elif normalization_scheme == 'power_transformation_bc':
            transformer = PowerTransformer(method="box-cox").fit(train_data)
            train_data, validation_data, test_data, anomalous_data = transformer.transform(train_data), transformer.transform(validation_data), transformer.transform(test_data), transformer.transform(anomalous_data)
        elif normalization_scheme == 'max_abs_scaling':
            transformer = MaxAbsScaler().fit(train_data)
            train_data, validation_data, test_data, anomalous_data = transformer.transform(train_data), transformer.transform(validation_data), transformer.transform(test_data), transformer.transform(anomalous_data)
        elif normalization_scheme == 'log_scaling':
            train_data, validation_data, test_data, anomalous_data = log_scaling(train_data), log_scaling(validation_data), log_scaling(test_data), log_scaling(anomalous_data)
        else:
            transformer = StandardScaler().fit(train_data)
            train_data, validation_data, test_data, anomalous_data = transformer.transform(train_data), transformer.transform(validation_data), transformer.transform(test_data), transformer.transform(anomalous_data)
            #full_data = np.concatenate((train_data, validation_data, test_data,anomalous_data), axis=0)
            #mu, sd = np.mean(train_data, axis=0, keepdims=True), np.std(train_data, axis=0, keepdims=True) + eps
            #train_data, validation_data, test_data, anomalous_data = (train_data - mu) / sd, (validation_data - mu) / sd, (test_data - mu) / sd, (anomalous_data - mu) / sd
    if removeNoise:
        # remove noise in data by simply checking their standard score
        NUM_FEATURE = train_data.shape[1]
        if False:
            not_noise = abs(train_data) < noise_threshold
            idc = np.sum(not_noise, axis=1) >= NUM_FEATURE
            train_data = train_data[idc,]
        else:
            eps = 1e-8
            mu, sd = np.mean(train_data, axis=0, keepdims=True), np.std(train_data, axis=0, keepdims=True) + eps
            X = (train_data - mu) / sd
            not_noise = abs(X) < noise_threshold
            idc = np.sum(not_noise, axis=1) >= NUM_FEATURE
            noise_ratio = 1 - idc.sum()/len(train_data)
            train_data = train_data[idc,]
        print(f"removeNoise = True ---- remaining Data Shape = {train_data.shape}; noise proportion: {noise_ratio}")
    print('Train data shape after normalize: ', train_data.shape)
    return train_data, validation_data, test_data, anomalous_data

def preProcessData_LogisticRegression(raw_data:np.array, anomalous_data:np.array, split:tuple or list=(0.6, 0.2, 0.2), trim:bool=False, trim_threshold:float=0.96, normalize:bool=True, cross_validation:bool=True, filterLinearDependencies:bool = True, filter_threshold:float=0.95, removeNoise:bool=False, noise_threshold:float=3, train_proportion=0.8, normalization_scheme='standard_scaling', categorical_data_index=None, over_sampling=True):   
    train_data, validation_data, test_data, a_data = preProcessData_OneClass(raw_data, anomalous_data, split=split, trim=trim, trim_threshold=trim_threshold, normalize=normalize, normalization_scheme=normalization_scheme, cross_validation=cross_validation, filterLinearDependencies = filterLinearDependencies, filter_threshold=filter_threshold, removeNoise=removeNoise, noise_threshold=noise_threshold, categorical_data_index=categorical_data_index)
    train_data = np.concatenate((train_data, validation_data, test_data), axis=0)
    train_data = np.concatenate((train_data, np.zeros((train_data.shape[0], 1))), axis=1)
    a_data = np.concatenate((a_data, np.ones((a_data.shape[0], 1))), axis=1)
    raw_data = np.concatenate((train_data, a_data), axis=0)
    random_idc = np.random.choice(len(raw_data), len(raw_data), replace=False)
    raw_data = raw_data[random_idc,:]
    #
    train_X, train_Y = raw_data[0:math.floor(len(raw_data) * train_proportion)][:,:-1], raw_data[0:math.floor(len(raw_data) * train_proportion)][:,-1:]
    test_X, test_Y = raw_data[math.floor(len(raw_data) * train_proportion):][:,:-1], raw_data[math.floor(len(raw_data) * train_proportion):][:,-1:]
    # resample
    if over_sampling:
        ros = RandomOverSampler(random_state=0)
        train_X, train_Y = ros.fit_resample(train_X, train_Y)
        train_Y = train_Y.reshape((train_Y.shape[0], 1))
    print("train_X, train_Y shape:", train_X.shape, train_Y.shape)
    print("test_X, test_Y shape:", test_X.shape, test_Y.shape)
    return train_X, train_Y, test_X, test_Y

def train(model:torch.nn.Module, optimization:str, epochs:int, train_X:torch.tensor, train_Y:torch.tensor, 
                criterion:torch.nn.modules.loss, dsvdd_y=None, batch_size:int=256, lr:float=1e-4, weight_decay:float=0, grad_limit=1e3, att:torch.nn.Module=None):
    model.train()
    loss_list = []
    train_data = train_X
    if optimization == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else: # for now, else use SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if att:
        optimizer_att = torch.optim.SGD(att.parameters(), lr=lr, weight_decay=weight_decay)
        att.initAllWeights()
    curr_time = time.time()
    for epoch in range(epochs):
        if att:
            optimizer_att.zero_grad()
            train_data = att(train_X)
            if model.name != 'Discriminator':
                train_Y = train_data.clone()
        if model.name == 'DAE':
            if att:
                # construct noisy train data
                train_mu, train_std = torch.mean(train_data, dim=0, keepdim=True), torch.std(train_data, dim=0, unbiased=True, keepdim=True)
                normalized = (train_data - train_mu) / (train_std + 1e-5)
                # assuming that the data is normalized
                train_data = train_data + model.noise_factor * torch.nn.functional.dropout(torch.rand_like(train_data), p=1-model.noise_fraction) * train_std
                #train_data = noisy_train_data.to(torch.float32).to(device)
            else:
                # construct noisy train data
                train_mu, train_std = torch.mean(train_X, dim=0, keepdim=True), torch.std(train_X, dim=0, unbiased=True, keepdim=True)
                normalized = (train_X - train_mu) / (train_std + 1e-8)
                # assuming that the data is normalized
                train_data = train_X + model.noise_factor * torch.nn.functional.dropout(torch.rand_like(train_X), p=1-model.noise_fraction) * train_std / torch.exp((normalized)**2)
                #train_data = noisy_train_data.to(torch.float32).to(device)
        loss, l = 0, len(train_data)
        permutation = np.random.permutation(l)
        for i in range(0, l, batch_size):
                
            batch_idc = permutation[i:i+batch_size]
            batch_X = train_data[batch_idc,]
            batch_Y = train_Y[batch_idc,]
            
            optimizer.zero_grad()
            # compute reconstructions
            if model.name == 'DSVDD' and dsvdd_y != None:
                #mu, sd = args[0], args[1]
                #y = torch.normal(mu.expand(math.floor(batch_size/20), model.num_feature), sd.expand(math.floor(batch_size/20), model.num_feature))
                y = dsvdd_y[batch_idc,]
                outputs, _ = model(batch_X, y)
            else:
                outputs = model(batch_X) 
            # compute training reconstruction loss
            sm = nn.Softmax(dim=-1)
            train_loss = criterion(outputs, batch_Y)
            #train_loss = torch.sum(torch.sum((outputs-batch_Y)**2))
            if model.error:
                train_loss += model.error
            """ if model.name == 'VAE':
                # add KL loss if model is Variational Auto Encoder
                train_loss += model.sigma * model.KLD.sum()
            if model.name == 'DOCAE' or model.name == 'DCOCAE' or model.name == 'DSVDE':
                # add error if model is Deep One-Class Auto Encoder/DSVDD
                train_loss += model.error
            if model.name == 'DSVDD':
                train_loss = 0
                train_loss = model.error """
            
            if att:
                for param in model.parameters():
                    train_loss += criterion(param, torch.zeros_like(param))
                train_loss.backward(retain_graph=True)
            elif dsvdd_y != None:
                train_loss.backward(retain_graph=True)
            else:
                train_loss.backward(retain_graph=False)
            loss += train_loss.item()
            # compute accumulated gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_limit)
            # perform parameter update based on current gradients
            optimizer.step()

        if att:
            torch.nn.utils.clip_grad_norm_(att.parameters(), grad_limit*1e-1)
            optimizer_att.step()
        loss /= l
        loss_list.append(loss)
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
    print("Train time: {:.4f} seconds.".format(time.time()-curr_time))
    return loss_list

def validate():
    pass

def test(model:torch.nn.Module, criterion:torch.nn.modules.loss, train_data:torch.Tensor, test_data:torch.Tensor, anomalous_data:torch.Tensor):
    # start eval mode
    model.eval()
    # calculate loss contribution score for training(normal) data
    difference = torch.square(model(train_data) - train_data)
    lcs_mean = torch.mean(torch.div(difference, torch.sum(difference, 1, True)), 0, True)
    
    i = 1
    path = '../LCS'
    filename = model.name
    pathfile = os.path.normpath(os.path.join(path, filename))
    if not os.path.exists(path):
        os.makedirs(path)
    while os.path.isfile(pathfile + '.txt'):
        pathfile = os.path.normpath(os.path.join(path, filename + str(i)))
        i += 1
    np.savetxt(pathfile + '.txt', lcs_mean.detach().numpy())
    
    num_feature, loss_test, loss_attack, kl_div_test, kl_div_attack, y_scores_loss, y_scores_lcs, y_ground_truth = model.num_feature, np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    lcs_array_test = torch.tensor([[0 for _ in range(model.num_feature)]])
    # Define KL Loss
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)

    l = len(test_data)
    permutation = np.random.permutation(l)
    for i in range(0, l, 1):
        # shuffle batch
        batch_idc = permutation[i:i+1]
        batch = test_data[batch_idc,]
        Y, loss = batch, 0
        # compute reconstructions
        outputs = model(batch)
        # compute the epoch test loss
        if model.error:
            loss = (criterion(outputs, Y) + model.error).item()
        else:
            loss = criterion(outputs, Y).item()
        # append loss and class label
        y_scores_loss = np.append(y_scores_loss, loss) # for ROC and AUC plotting
        y_ground_truth = np.append(y_ground_truth, 0)
        loss_test = np.append(loss_test, loss)
        # compute loss contribution score
        lcs = (torch.square(outputs - Y) / loss).view(1, num_feature)
        # KL Divergence
        kl_div_test = np.append(kl_div_test, kl_loss(torch.log(lcs), lcs_mean.view(1, num_feature)).detach().numpy())
        lcs_array_test = torch.cat((lcs_array_test, lcs), 0)

    lcs_array_test = lcs_array_test[1:].detach()
    # lcs_mean_test = torch.mean(lcs_array_test, dim=0)
    # Loss Contribution Score
    y_scores_lcs = torch.sum((torch.square(lcs_array_test - lcs_mean)), dim=1, keepdim=False)
    lcs_array_test = lcs_array_test.numpy()
    #________________________________________________________________________#

    # anomaly detection test
    lcs_array_attack = torch.tensor([[0 for _ in range(num_feature)]])
    l = len(anomalous_data)
    permutation = np.random.permutation(l)
    for i in range(0, l, 1):
        # shuffle batch
        batch_idc = permutation[i:i+1]
        batch = anomalous_data[batch_idc,]
        loss, Y = 0, batch
        outputs = model(batch) # compute reconstructions
        loss = criterion(outputs, Y).item() # compute loss
        # append loss and class label
        y_scores_loss = np.append(y_scores_loss, loss) # for ROC and AUC plotting
        y_ground_truth = np.append(y_ground_truth, 1)
        loss_attack = np.append(loss_attack, loss)
        lcs = torch.square(outputs - Y) / loss
        # KL Divergence
        kl_div_attack = np.append(kl_div_attack, kl_loss(torch.log(lcs), lcs_mean.view(1, num_feature)).detach().numpy())
        lcs_array_attack = torch.cat((lcs_array_attack, torch.reshape(lcs, (1, num_feature))), 0)

    lcs_array_attack = lcs_array_attack[1:].detach()
    #lcs_mean_attack = torch.mean(lcs_array_attack, dim=0)
    # Loss Contribution Score
    y_scores_lcs = torch.cat((y_scores_lcs, torch.sum((torch.square(lcs_array_attack - lcs_mean)), dim=1, keepdim=False))).detach().numpy()
    lcs_array_attack = lcs_array_attack.numpy()
    return loss_test, loss_attack, kl_div_test, kl_div_attack, y_scores_loss, y_scores_lcs, y_ground_truth, lcs_array_test, lcs_array_attack

def test_light(model:torch.nn.Module, criterion:torch.nn.modules.loss, train_data:torch.Tensor, test_data:torch.Tensor, anomalous_data:torch.Tensor):
    # start eval mode
    model.eval()
    #loss_test, loss_attack = np.array([]), np.array([])
    original_reduction = criterion.reduction
    criterion.reduction = 'none'
    # test loss
    loss_test = torch.sum(criterion(model(test_data), test_data), dim=-1)
    loss_test = loss_test.detach().numpy()
    # attack loss
    loss_attack = torch.sum(criterion(model(anomalous_data), anomalous_data), dim=-1)
    loss_attack = loss_attack.detach().numpy()

    criterion.reduction = original_reduction
    return loss_test, loss_attack

def test_classification(model:torch.nn.Module, criterion:torch.nn.modules.loss, test_X:torch.Tensor, test_Y:torch.Tensor):
    # start eval mode
    model.eval()
    #y_scores_loss, y_ground_truth = criterion(model(test_X), test_Y).detach().numpy(), test_Y.detach().numpy()
    y_scores_loss, y_ground_truth = model(test_X).detach().numpy(), test_Y.detach().numpy()
    return y_scores_loss, y_ground_truth

def visualize_convergence(loss_array,  model_name:str, save:bool=False, save_path:str="../graphs/Convergence", **kwargs):
    fig = plt.figure(figsize = (5, 5))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Epoch', fontsize = 12)
    ax.set_ylabel('Loss--L2 Distance', fontsize = 12)
    ax.set_yscale('log')
    ax.set_title('Convergence rate: ' + model_name, fontsize = 18)
    ax.grid()
    ax.plot(loss_array)
    # annotate graph
    x, y, i = 1.5, 1, 1
    for key, value in kwargs.items():
        ax.text(x, y, str(key) + " = " + str(value), ha='right', va='top', transform = ax.transAxes)
        y -= 0.05
    # save graph
    if save:
        path = save_path
        filename = model_name
        pathfile = os.path.normpath(os.path.join(path, filename))
        if not os.path.exists(path):
            os.makedirs(path)
        while os.path.isfile(pathfile + '.png'):
            pathfile = os.path.normpath(os.path.join(path, filename + str(i)))
            i += 1
        fig.savefig(pathfile, bbox_inches='tight')
    return ax
    
def visualize_loss(loss_test, loss_attack, model_name:str, save:bool=False, save_path:str="../graphs/loss", **kwargs):
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Instances', fontsize = 12)
    ax.set_ylabel('MSE---Sum', fontsize = 12)
    ax.set_yscale('log')
    ax.set_title('Reconstruction Error: ' + model_name, fontsize = 18)
    ax.grid()
    ax.scatter(np.arange(len(loss_test)), loss_test, marker=".", alpha=0.2).set_label('Normal data')
    ax.scatter(np.arange(len(loss_attack)), loss_attack, marker="x", alpha=0.1).set_label('Anomalous data')
    ax.legend()
    # annotate graph
    x, y, i = 1.5, 1, 1
    for key, value in kwargs.items():
        ax.text(x, y, str(key) + " = " + str(value), ha='right', va='top', transform = ax.transAxes)
        y -= 0.05
    # save graph
    if save:
        path = save_path
        filename = model_name
        pathfile = os.path.normpath(os.path.join(path, filename))
        if not os.path.exists(path):
            os.makedirs(path)
        while os.path.isfile(pathfile + '.png'):
            pathfile = os.path.normpath(os.path.join(path, filename + str(i)))
            i += 1
        fig.savefig(pathfile, bbox_inches='tight')
    return ax

def visualize_kl(kl_div_test, kl_div_attack, model_name:str, save:bool=False, save_path:str="../graphs/KL_DIV", **kwargs):
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(np.arange(len(kl_div_test)), kl_div_test, marker=".").set_label('Normal data')
    ax.scatter(np.arange(len(kl_div_attack)), kl_div_attack, marker="x", alpha=0.6).set_label('Anomalous data')
    ax.set_title('KL-DIV with average LCS of train data: ' + model_name, fontsize = 18)
    ax.set_xlabel('Instances', fontsize = 12)
    ax.set_ylabel('KL-DIV', fontsize = 12)
    ax.legend()
    # annotate graph
    x, y, i = 1.5, 1, 1
    for key, value in kwargs.items():
        ax.text(x, y, str(key) + " = " + str(value), ha='right', va='top', transform = ax.transAxes)
        y -= 0.05
    # save graph
    if save:
        path = save_path
        filename = model_name
        pathfile = os.path.normpath(os.path.join(path, filename))
        if not os.path.exists(path):
            os.makedirs(path)
        while os.path.isfile(pathfile + '.png'):
            pathfile = os.path.normpath(os.path.join(path, filename + str(i)))
            i += 1
        fig.savefig(pathfile, bbox_inches='tight')
    return ax

def visualize_tSNE(tsne_code, len_test_data, len_anomalous_data, perplexity, model_name:str, save:bool=False, save_path:str="../graphs/tSNE/latent_feature", **kwargs):
    fig1 = plt.figure(figsize = (6,6))
    ax1 = fig1.add_subplot(1,1,1)
    l1, l2 = len_test_data, len_anomalous_data
    ax1.scatter(tsne_code[:l1,0], tsne_code[:l1,1], marker=".").set_label('Normal data')
    ax1.scatter(tsne_code[l1:,0], tsne_code[l1:,1], marker="x", alpha=0.4).set_label('Anomalous data')
    ax1.set_title(('tSNE of codes (perplexity = ' + str(perplexity) + '): ' + model_name), fontsize = 18)
    ax1.set_xlabel('tSNE-1', fontsize = 12)
    ax1.set_ylabel('tSNE-2', fontsize = 12)
    ax1.legend()
    # annotate graph
    x, y, i = 1.5, 1, 1
    for key, value in kwargs.items():
        ax1.text(x, y, str(key) + " = " + str(value), ha='right', va='top', transform = ax1.transAxes)
        y -= 0.05
    # save graph
    if save:
        path = save_path
        filename = model_name
        pathfile = os.path.normpath(os.path.join(path, filename))
        if not os.path.exists(path):
            os.makedirs(path)
        while os.path.isfile(pathfile + '.png'):
            pathfile = os.path.normpath(os.path.join(path, filename + str(i)))
            i += 1
        fig1.savefig(pathfile, bbox_inches='tight')
    
    return ax1

def visualize_ROC(y_ground_truth, model_name:str, save:bool, scores, save_path:str="../graphs/ROC", **kwargs):
    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(1,1,1) 
    ax.set_title(model_name + '(normalized data)\n ROC Curve and AUC', fontsize = 18)
    for key, score in scores.items():
        fpr, tpr, _ = metrics.roc_curve(y_ground_truth, score)
        auc = metrics.roc_auc_score(y_ground_truth, score)
        ax.plot(fpr, tpr, label= str(key) + " auc = "+str(auc))
    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_ylabel("True Positive Rate", fontsize=15)
    ax.legend(loc=4)
    # annotate graph
    x, y, i = 1.5, 1, 1
    for key, value in kwargs.items():
        ax.text(x, y, str(key) + " = " + str(value), ha='right', va='top', transform = ax.transAxes)
        y -= 0.05
    # save graph
    if save:
        path = save_path
        filename = model_name
        pathfile = os.path.normpath(os.path.join(path, filename))
        if not os.path.exists(path):
            os.makedirs(path)
        while os.path.isfile(pathfile + '.png'):
            pathfile = os.path.normpath(os.path.join(path, filename + str(i)))
            i += 1
        fig.savefig(pathfile, bbox_inches='tight')
    
def visualize_curve(metrics, save:bool=False, title:str='', x_label='', y_label='', save_path:str="../graphs/curve", **kwargs):
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1)
    
    ax.set_title(title, fontsize = 18)
    for key, (x, y) in metrics.items():
        # smooth curve
        ax.plot(x, y, label= str(key), alpha=0.5,)
    
    plt.xticks(np.arange(0,1.05, step=0.1))
    plt.yticks(np.arange(0,1.05, step=0.1))
    plt.grid(visible=True)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    
    ax.legend(loc=4)
    # annotate graph
    x, y, i = 1.5, 1, 1
    for key, value in kwargs.items():
        ax.text(x, y, str(key) + " = " + str(value), ha='right', va='top', transform = ax.transAxes)
        y -= 0.05
    # save graph
    if save:
        path = save_path
        filename = title
        pathfile = os.path.normpath(os.path.join(path, filename))
        if not os.path.exists(path):
            os.makedirs(path)
        while os.path.isfile(pathfile + '.png'):
            pathfile = os.path.normpath(os.path.join(path, filename + str(i)))
            i += 1
        fig.savefig(pathfile, bbox_inches='tight')

def visualize_bar(data:dict, save:bool=False, title:str='Bar Plot', save_path:str="../graphs/bar_plot", **kwargs):
    fig = plt.figure(figsize = (7,7))
    ax = fig.add_subplot(1,1,1)
    ax.set_title(title, fontsize = 18)
    x, y = list(data.keys()), list(data.values())
    ax.barh(x, y, 
            #color=['black', 'red', 'green', 'blue', 'cyan', 'brown', 'yellow', 'pink', 'purple', 'tan']
            )
    ax.set_xscale('log')
    # show values
    """ for index, value in enumerate(y):
        ax.text(value, index, str(value)[:6], horizontalalignment='left') """
    i = 1
    if save:
        path = save_path
        filename = title
        pathfile = os.path.normpath(os.path.join(path, filename))
        if not os.path.exists(path):
            os.makedirs(path)
        while os.path.isfile(pathfile + '.png'):
            pathfile = os.path.normpath(os.path.join(path, filename + str(i)))
            i += 1
        fig.savefig(pathfile, bbox_inches='tight')
   
def plot_bar(data:dict, model_name:str, save:bool=False, save_path='../graphs/sensitivity_specificity', **kwargs):
    items, values = list(data.keys()), list(data.values())
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1)
    ax.bar(items, values, width=0.5)
    ax.set_title("Hypersphere classification capability: " + model_name)
    ax.set_xlabel("Criterions")
    ax.set_ylabel("Percentage")
    # annotate graph
    x, y, i = 1.5, 1, 1
    for key, value in kwargs.items():
        ax.text(x, y, str(key) + " = " + str(value), ha='right', va='top', transform = ax.transAxes)
        y -= 0.05
    # save graph
    if save:
        path = save_path
        filename = model_name
        pathfile = os.path.normpath(os.path.join(path, filename))
        if not os.path.exists(path):
            os.makedirs(path)
        while os.path.isfile(pathfile + '.png'):
            pathfile = os.path.normpath(os.path.join(path, filename + str(i)))
            i += 1
        fig.savefig(pathfile, bbox_inches='tight')
    
def toTorchTensor(device, *args):
    res = []
    for arr in args:
        res.append(torch.tensor(arr).to(torch.float).to(device))
    return (arr for arr in res)

def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)

def SNIP(net, keep_ratio, train_dataloader, device):
    # TODO: shuffle?

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(net)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = F.nll_loss(outputs, targets)
    loss.backward()

    grads_abs = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads_abs.append(torch.abs(layer.weight_mask.grad))

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

    return(keep_masks)

def log_scaling(x:torch.tensor):
    eps = 1e-8
    res = np.abs(np.log(np.abs(x) + eps))
    res = res / np.sum(res, axis=-1, keepdims=True)
    return np.log(res + eps)

def getLoss(model, criterion, *args, weights=None, Gaussian=False):
    if weights is not None:
        if Gaussian:
            std = torch.diag(weights).view(1, len(weights)) # 1 x d row vector
            return (torch.sum(criterion(model(d), d) * (1.65**((criterion(model(d), d)/std)**2)), dim=-1, keepdim=False) for d in args)
        return ((torch.sum(torch.matmul(criterion(model(d), d), weights), dim=-1, keepdim=False)) for d in args)
    else:
        return ((torch.sum(criterion(model(d), d), dim=-1, keepdim=False)) for d in args)

def MCC(specificity, sensitivity):
    TN, FN, TP, FP = specificity, 1-specificity, sensitivity, 1-sensitivity
    return (TP*TN - FP*FN) / ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5

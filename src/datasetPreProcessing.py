import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer, make_column_selector

# data loader functions and data preprocessing functions
def LoadDatasetByName(DATASET, *args):
    raw_data, anomalous_raw, features, categorical_data_index = None, None, None, None
    if DATASET == 'original_data':
        raw_data = np.loadtxt('../data/original_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/original_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
        categorical_data_index = [0]
    if DATASET == 'network_data':
        raw_data = np.loadtxt('../data/network_flow_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/network_flow_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
        """ idc = np.bool_(np.loadtxt('../data/idc/data_idc1.csv', delimiter=','))
        anomalous_raw = anomalous_raw[idc,:]
        idc = np.bool_(np.loadtxt('../data/idc/data_idc2.csv', delimiter=','))
        anomalous_raw = anomalous_raw[idc,:] """
    if DATASET == 'network_data_h':
        #h_indices = [2,4,5,6,7,8]
        raw_data = np.loadtxt('../data/network_flow_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/network_flow_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
        #anomalous_raw = np.loadtxt('../data/network_flow_attack_data_undetected.csv', delimiter=',')[:,:-1]
    elif DATASET == 'medical_data':
        raw_data = np.loadtxt('../data/medical_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/medical_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
        idc = np.bool_(np.loadtxt('../data/idc/data_idc1.csv', delimiter=','))
        anomalous_raw = anomalous_raw[idc,:]
        #anomalous_raw = np.loadtxt('../data/medical_attack_data_undetected.csv', skiprows=1, delimiter=',')[:,:-1]
    elif DATASET == 'medical_full_data':
        raw_data = np.loadtxt('../data/medical_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/medical_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
    elif DATASET == 'full_data':
        raw_data = np.concatenate((np.loadtxt('../data/network_flow_regular_data.csv', skiprows=1, delimiter=',')[:,:-1], np.loadtxt('../data/medical_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]), axis=1)
        anomalous_raw = np.concatenate((np.loadtxt('../data/network_flow_attack_data.csv', skiprows=1, delimiter=',')[:,:-1], np.loadtxt('../data/medical_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]), axis=1)
        features = np.concatenate((np.loadtxt('../data/network_flow_regular_data.csv', max_rows=1, delimiter=',', dtype=str), np.loadtxt('../data/medical_regular_data.csv', max_rows=1, delimiter=',', dtype=str)), axis=0)
        """ idc = np.bool_(np.loadtxt('../data/idc/data_idc1.csv', delimiter=','))
        anomalous_raw = anomalous_raw[idc,:]
        idc = np.bool_(np.loadtxt('../data/idc/data_idc2.csv', delimiter=','))
        anomalous_raw = anomalous_raw[idc,:] """
    elif DATASET == 'network_data1':
        raw_data = np.loadtxt('../data/network_flow1_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/network_flow1_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
    elif DATASET == 'medical_data1':
        #raw_data = np.loadtxt('../data/medical_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        raw_data = np.loadtxt('../data/medical1_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        #raw_data = np.concatenate((np.loadtxt('../data/medical_regular_data.csv', skiprows=1, delimiter=',')[:,:-1], np.loadtxt('../data/medical1_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]), axis=0)
        anomalous_raw = np.loadtxt('../data/medical_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
        #anomalous_raw = np.loadtxt('../data/medical_attack_data_undetected.csv', skiprows=1, delimiter=',')[:,:-1]
    elif DATASET == 'full_data1':
        raw_data = np.concatenate((np.loadtxt('../data/network_flow1_regular_data.csv', skiprows=1, delimiter=',')[:,:-1], np.loadtxt('../data/medical1_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]), axis=1)
        anomalous_raw = np.concatenate((np.loadtxt('../data/network_flow1_attack_data.csv', skiprows=1, delimiter=',')[:,:-1], np.loadtxt('../data/medical1_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]), axis=1)
    elif DATASET == 'network_data2':
        raw_data = np.loadtxt('../data/network_flow2_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/network_flow2_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
    elif DATASET == 'iiot_data':
        raw_data = np.loadtxt('../data/iiot_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/iiot_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
        features = np.loadtxt('../data/iiot_attack_data.csv', max_rows=1, delimiter=',', dtype=str)
    elif DATASET == 'iiot_data_h':
        h_indices = [7,6,9,10,21,22,19,20,24,5,8,11,14]
        raw_data = np.loadtxt('../data/iiot_regular_data.csv', skiprows=1, delimiter=',')[:,h_indices]
        anomalous_raw = np.loadtxt('../data/iiot_attack_data.csv', skiprows=1, delimiter=',')[:,h_indices]
        features = np.loadtxt('../data/iiot_attack_data.csv', max_rows=1, delimiter=',', dtype=str)
    elif DATASET == 'ECU_IoHT_data':
        raw_data = np.loadtxt('../data_backup/ECU_IoHT_regular_data.csv', skiprows=1, delimiter=',', dtype=str)[:,1:-2]
        anomalous_raw = np.loadtxt('../data_backup/ECU_IoHT_attack_data.csv', skiprows=1, delimiter=',', dtype=str)[:,1:-2]
        features = np.loadtxt('../data_backup/ECU_IoHT_attack_data.csv', max_rows=1, delimiter=',', dtype=str)
        categorical_data_index = [0,1,2]
    elif DATASET == 'ECU_IoHT_data_t':
        raw_data = np.loadtxt('../data_backup/ECU_IoHT_regular_data.csv', skiprows=1, delimiter=',', dtype=str)[:,3:-2]
        anomalous_raw = np.loadtxt('../data_backup/ECU_IoHT_attack_data.csv', skiprows=1, delimiter=',', dtype=str)[:,3:-2]
        features = np.loadtxt('../data_backup/ECU_IoHT_attack_data.csv', max_rows=1, delimiter=',', dtype=str)
        categorical_data_index = [0]
    elif DATASET == 'ECU_IoHT_data_d':
        raw_data = np.loadtxt('../data_backup/ECU_IoHT_regular_data.csv', skiprows=1, delimiter=',', dtype=str)[:,2:-2]
        anomalous_raw = np.loadtxt('../data_backup/ECU_IoHT_attack_data.csv', skiprows=1, delimiter=',', dtype=str)[:,2:-2]
        features = np.loadtxt('../data_backup/ECU_IoHT_attack_data.csv', max_rows=1, delimiter=',', dtype=str)
        categorical_data_index = [0, 1]
    elif DATASET == 'creditcard_data':
        raw_data = np.loadtxt('../data/creditcard_regular_data.csv', skiprows=1, delimiter=',', usecols=(i for i in range(30)))[:,1:]
        anomalous_raw = np.loadtxt('../data/creditcard_attack_data.csv', skiprows=1, delimiter=',', usecols=(i for i in range(30)))[:,1:]
        #features = np.loadtxt('../data/creditcard_attack_data.csv', max_rows=1, delimiter=',', dtype=str)
    elif DATASET == 'TON_IoT_data':
        index = [0,1,2,6,12,13,14,17]
        index_used = [2,3,4,5,7,8,9,10,11,15,16]
        #raw_data = np.loadtxt('../data/TON_IoT_regular_data.csv', skiprows=1, delimiter=',', dtype=str, usecols=index_used)[:,:-2]
        #anomalous_raw = np.loadtxt('../data/TON_IoT_attack_data.csv', skiprows=1, delimiter=',', dtype=str, usecols=index_used)[:,:-2]
        raw_data = np.loadtxt('../data/TON_IoT_regular_data.csv', skiprows=1, delimiter=',', dtype=str, usecols=index_used)
        anomalous_raw = np.loadtxt('../data/TON_IoT_attack_data.csv', skiprows=1, delimiter=',', dtype=str, usecols=index_used)
        categorical_data_index = [0]
        #features = np.loadtxt('../data/TON_IoT_data_attack_data.csv', max_rows=1, delimiter=',', dtype=str)
    elif DATASET == 'TON_IoT_full_data':
        index_used = [i for i in range(2, 18)]
        #raw_data = np.loadtxt('../data/TON_IoT_regular_data.csv', skiprows=1, delimiter=',', dtype=str, usecols=index_used)[:,:-2]
        #anomalous_raw = np.loadtxt('../data/TON_IoT_attack_data.csv', skiprows=1, delimiter=',', dtype=str, usecols=index_used)[:,:-2]
        raw_data = np.loadtxt('../data/TON_IoT_regular_data.csv', skiprows=1, delimiter=',', dtype=str, usecols=index_used)
        anomalous_raw = np.loadtxt('../data/TON_IoT_attack_data.csv', skiprows=1, delimiter=',', dtype=str, usecols=index_used)
        categorical_data_index = [0,4,10,11,12,15]
    elif DATASET == 'TON_IoT_full_data_t':
        index_used = [i for i in range(2, 18)]
        raw_data = np.loadtxt('../data/TON_IoT_regular_data.csv', skiprows=1, delimiter=',', dtype=str, usecols=index_used)
        anomalous_raw = np.loadtxt('../data/TON_IoT_attack_data.csv', skiprows=1, delimiter=',', dtype=str, usecols=index_used)
        categorical_data_index = [0,4,10,11,12,15]
    elif DATASET == 'KDD_10':
        raw_data = np.loadtxt('../data/KDD_10_regular_data.csv', skiprows=1, delimiter=',')
        anomalous_raw = np.loadtxt('../data/KDD_10_attack_data.csv', skiprows=1, delimiter=',')
    # return loaded data
    return raw_data, anomalous_raw, features, categorical_data_index

def OneHotTransformer(X, index):
    ohe = OneHotEncoder(categories='auto', handle_unknown='infrequent_if_exist', dtype=np.float64, sparse_output=False)
    column_transformer = make_column_transformer(
        (ohe, index),
        remainder='passthrough'
    )
    column_transformer.fit(X)
    return column_transformer


def LoadFeatureByName(DATASET, *args):
    raw_data, anomalous_raw, features, categorical_data_index = None, None, None, None
    if DATASET == 'original_data':
        feature = np.loadtxt('../data/original_regular_data.csv', max_rows=1, dtype=str, delimiter=',')[:,:-1]
    if DATASET == 'network_data':
        feature = np.loadtxt('../data/network_flow_regular_data.csv', max_rows=1, dtype=str, delimiter=',')[:,:-1]
    if DATASET == 'network_data_h':
        #h_indices = [2,4,5,6,7,8]
        raw_data = np.loadtxt('../data/network_flow_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/network_flow_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
        #anomalous_raw = np.loadtxt('../data/network_flow_attack_data_undetected.csv', delimiter=',')[:,:-1]
    elif DATASET == 'medical_data':
        raw_data = np.loadtxt('../data/medical_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/medical_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
        idc = np.bool_(np.loadtxt('../data/idc/data_idc1.csv', delimiter=','))
        anomalous_raw = anomalous_raw[idc,:]
        #anomalous_raw = np.loadtxt('../data/medical_attack_data_undetected.csv', skiprows=1, delimiter=',')[:,:-1]
    elif DATASET == 'medical_full_data':
        raw_data = np.loadtxt('../data/medical_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/medical_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
    elif DATASET == 'full_data':
        feature = np.concatenate((np.loadtxt('../data/network_flow_regular_data.csv', max_rows=1, dtype=str, delimiter=',')[:-1], np.loadtxt('../data/medical_regular_data.csv', max_rows=1, dtype=str, delimiter=',')[:-1]), axis=-1)
    elif DATASET == 'network_data1':
        raw_data = np.loadtxt('../data/network_flow1_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/network_flow1_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
    elif DATASET == 'medical_data1':
        #raw_data = np.loadtxt('../data/medical_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        raw_data = np.loadtxt('../data/medical1_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        #raw_data = np.concatenate((np.loadtxt('../data/medical_regular_data.csv', skiprows=1, delimiter=',')[:,:-1], np.loadtxt('../data/medical1_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]), axis=0)
        anomalous_raw = np.loadtxt('../data/medical_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
        #anomalous_raw = np.loadtxt('../data/medical_attack_data_undetected.csv', skiprows=1, delimiter=',')[:,:-1]
    elif DATASET == 'full_data1':
        raw_data = np.concatenate((np.loadtxt('../data/network_flow1_regular_data.csv', skiprows=1, delimiter=',')[:,:-1], np.loadtxt('../data/medical1_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]), axis=1)
        anomalous_raw = np.concatenate((np.loadtxt('../data/network_flow1_attack_data.csv', skiprows=1, delimiter=',')[:,:-1], np.loadtxt('../data/medical1_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]), axis=1)
    elif DATASET == 'network_data2':
        raw_data = np.loadtxt('../data/network_flow2_regular_data.csv', skiprows=1, delimiter=',')[:,:-1]
        anomalous_raw = np.loadtxt('../data/network_flow2_attack_data.csv', skiprows=1, delimiter=',')[:,:-1]
    elif DATASET == 'iiot_data':
        feature = np.loadtxt('../data/iiot_attack_data.csv', max_rows=1, delimiter=',', dtype=str)[:-1]
    elif DATASET == 'iiot_data_h':
        h_indices = [7,6,9,10,21,22,19,20,24,5,8,11,14]
        raw_data = np.loadtxt('../data/iiot_regular_data.csv', skiprows=1, delimiter=',')[:,h_indices]
        anomalous_raw = np.loadtxt('../data/iiot_attack_data.csv', skiprows=1, delimiter=',')[:,h_indices]
        feature = np.loadtxt('../data/iiot_attack_data.csv', max_rows=1, delimiter=',', dtype=str)
    elif DATASET == 'ECU_IoHT_data':
        feature = np.loadtxt('../data_backup/ECU_IoHT_attack_data.csv', max_rows=1, delimiter=',', dtype=str)
    elif DATASET == 'ECU_IoHT_data_t':
        feature = np.loadtxt('../data_backup/ECU_IoHT_attack_data.csv', max_rows=1, delimiter=',', dtype=str)[:,3:-2]
    elif DATASET == 'creditcard_data':
        raw_data = np.loadtxt('../data/creditcard_regular_data.csv', skiprows=1, delimiter=',', usecols=(i for i in range(30)))[:,1:]
        anomalous_raw = np.loadtxt('../data/creditcard_attack_data.csv', skiprows=1, delimiter=',', usecols=(i for i in range(30)))[:,1:]
        #features = np.loadtxt('../data/creditcard_attack_data.csv', max_rows=1, delimiter=',', dtype=str)
    elif DATASET == 'TON_IoT_data':
        feature = np.loadtxt('../data/TON_IoT_data_attack_data.csv', max_rows=1, delimiter=',', dtype=str)
    elif DATASET == 'TON_IoT_full_data':
        index_used = [i for i in range(2, 18)]
        feature = np.loadtxt('../data/TON_IoT_attack_data.csv', max_rows=1, delimiter=',', dtype=str, usecols=index_used)
    elif DATASET == 'KDD_10':
        feature = np.loadtxt('../data/KDD_10_attack_data.csv', max_rows=1, delimiter=',', dtype=str)
    return feature
    # return loaded data




import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import QuantileTransformer


ddos_model =  LinearSVC(C=1.0, max_iter=10000, verbose=True)
scaler = QuantileTransformer( random_state=42)

def run_ddos_detect(display_log= True, display_graphs= True ):
    print("Starting: File Read")
    data = pd.read_csv('datasets/final_dataset.csv')
    print("Finished: File Read")
    train_data, test_data = train_test_split(data,shuffle=True, test_size=0.25)
    # train and test model
    train(train_data)
    test(test_data)

def train(train_data):
    print("Starting: Training")
    # snag features
    flow_duration = train_data["Flow Duration"]
    fwd_pkts = train_data["Tot Fwd Pkts"]
    bwd_pkts =  train_data["Tot Bwd Pkts"]
    pkts = train_data["Flow Pkts/s"]
    byts = train_data["Flow Byts/s"]
    fwd_len_mean = train_data["Fwd Pkt Len Mean"]
    syn_cnt = train_data["SYN Flag Cnt"]
    idle_mean = train_data["Idle Mean"]

    X = np.column_stack([
        flow_duration,
        fwd_pkts,
        bwd_pkts,
        pkts,
        byts,
        fwd_len_mean,
        syn_cnt,
        idle_mean
    ])
    print("Finished: snagging features")
    # snag labels
    labels = train_data["Label"].to_numpy()

    # sanitize
    X = np.nan_to_num(X, nan=0.0, posinf=20000000.0, neginf=0.0)
    X = scaler.fit_transform(X)
    #do the randomforest
    print("Started: Fitting Model")
    ddos_model.fit(X, labels)
    print("Finished: Training")

def test(test_data):
    print("Starting: Testing")
    # snag features
    flow_duration = test_data["Flow Duration"]
    fwd_pkts = test_data["Tot Fwd Pkts"]
    bwd_pkts = test_data["Tot Bwd Pkts"]
    pkts = test_data["Flow Pkts/s"]
    byts = test_data["Flow Byts/s"]
    fwd_len_mean = test_data["Fwd Pkt Len Mean"]
    syn_cnt = test_data["SYN Flag Cnt"]
    idle_mean = test_data["Idle Mean"]
    X = np.column_stack([
        flow_duration,
        fwd_pkts,
        bwd_pkts,
        pkts,
        byts,
        fwd_len_mean,
        syn_cnt,
        idle_mean
    ])
    # snag labels
    labels = test_data["Label"].to_numpy()

    # sanitize
    X = np.nan_to_num(X, nan=0.0, posinf=20000000.0, neginf=0.0)
    X = scaler.transform(X)
    # do the model
    pred = ddos_model.predict(X)
    print("Classification Report (Testing Data):")
    print(classification_report(labels, pred))


    print("Finished: Testing")
run_ddos_detect()
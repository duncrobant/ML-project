from sklearn import preprocessing
import numpy as np
import pandas as pd

df = pd.read_csv('datasets/final_dataset.csv')

keep_features = [
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Flow Pkts/s", "Flow Byts/s", "Fwd Pkt Len Mean",
    "SYN Flag Cnt", "Idle Mean", "Label", "Src IP"
]
# Automatically find all columns you need to drop
drop_features = [col for col in df.columns if col not in keep_features]

print("Dropping these columns:")
for col in drop_features:
    print(col)

# Drop unwanted columns
df = df.drop(columns=drop_features)

df.to_csv('datasets/final_dataset.csv', index=False)



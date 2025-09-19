import pandas as pd
from sklearn.model_selection import train_test_split

file = pd.read_csv('./Datasets/Tg/Tg.csv')

train, test = train_test_split(file, test_size=0.2, random_state=42)

# save as txt files
train.to_csv('./NIPS_GNN/dataset/regression/Tg/data_train.txt', sep=" ", index=False, header=False)
test.to_csv('./NIPS_GNN/dataset/regression/Tg/data_test.txt', sep=" ", index=False, header=False)
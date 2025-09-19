## Data Sources:
- ### data/train.csv: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data?select=train.csv
- ### data/train_supplement/dataset3.csv: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data
- ### Tg_SMILES_class_pid_polyinfo_median.csv: https://github.com/Yeaaahhhhh/polyerdatasets/blob/main/Tg_SMILES_class_pid_polyinfo_median.csv
- ### tg_data_2: https://springernature.figshare.com/articles/dataset/dataset_with_glass_transition_temperature/24219958?file=42507037
- ### tg_data_1: https://www.sciencedirect.com/science/article/pii/S2590159123000377#ec0005
- ### pi.csv: https://github.com/hkqiu/DataAugmentation4SmallData/blob/main/model/training%20data/pi.csv
- ### polymer.csv: https://github.com/hkqiu/DataAugmentation4SmallData/blob/main/model/training%20data/polymer.csv
- ### tg_density.csv: https://www.kaggle.com/datasets/oleggromov/polymer-tg-density-excerpt || https://github.com/Shorku/rhnet/tree/main/data

## Steps:
- ### Download the above datasets.
- ### Run the `Datasets/Tg/data_augmentation.ipynb` file to get the `Tg.csv` augmented data.
- ### Or Use the already available `Tg.csv`.
- ### Use the NIPS GNN for Training & Inference on Tg Property.
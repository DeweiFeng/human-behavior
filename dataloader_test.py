from dataloaders.daic_loader import DAICWOZDataset
from torch.utils.data import Dataset, DataLoader

root_dir = "/home/dewei/workspace/dewei/dataset/daicwoz"

train_dataset = DAICWOZDataset(
    root_dir=root_dir,
    split_csv=root_dir + "/train_split_Depression_AVEC2017.csv",
    load_audio=False,
    load_covarep=True
)

dev_dataset = DAICWOZDataset(
    root_dir=root_dir,
    split_csv=root_dir + "/dev_split_Depression_AVEC2017.csv"
)

test_dataset = DAICWOZDataset(
    root_dir=root_dir,
    split_csv=root_dir + "/test_split_Depression_AVEC2017.csv"
)

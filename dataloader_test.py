from dataloaders.meld_loader import MELDDataset
from torch.utils.data import Dataset, DataLoader

dataset = MELDDataset(data_dir='/orcd/pool/003/dewei/dataset/meld/MELD.Raw', split='test')
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for batch in dataloader:
    videos = batch['video']
    labels = batch['label']
    filenames = batch['filename']

    print(type(videos))
    # do something here
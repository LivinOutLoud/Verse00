import torch, csv, os
import torch.nn.functional
from torch.utils.data import Dataset
from tqdm import tqdm
from data_utilities import *
import pandas as pd

modes_dict = {'all': ['training', 'validation', 'test'], '2020': ['01training', '02validation', '03test'], '2019': [
    '19training', '19validation', '19test']}


def load_rawdata(rootpath):
    for mode in tqdm(modes_dict['all']):
        csv_columns = ['ID', 'data', 'label']
        datadict_list = []
        for dir in os.listdir(rootpath):
            if dir.endswith(mode):
                for root, dirs, files in os.walk(os.path.join(rootpath, dir), topdown=True):
                    if 'derivatives' in root.split(os.sep):
                        data, label = '', ''
                        data_dict = {'ID': '', 'data': data, 'label': label}
                        for file in files:
                            if file.endswith('.nii.gz'):
                                data = os.path.join(root, file)
                            if file.endswith('.json'):
                                label = os.path.join(root, file)
                            data_dict = {'ID': file.split('.')[0], 'data': data, 'label': label}
                        if data_dict['ID'] != '' and data_dict['data'] != '' and data_dict['label'] != '':
                            datadict_list.append(data_dict)
                    else:
                        continue

            csvfile = mode + '.csv'
            with open(csvfile, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for dataitem_dict in datadict_list:
                    writer.writerow(dataitem_dict)


class VertebraCT(Dataset):
    def __init__(self, rootpath, mode):
        super(VertebraCT, self).__init__()
        self.rootpath = rootpath
        self.mode = mode
        self.images, self.labels, self.IDs = self.load_csv()

    def load_csv(self):
        images, labels, IDs = [], [], []
        df = pd.read_csv(self.mode + '.csv', dtype={'ID': str, 'data': str, 'label': str})
        for _, row in df.iterrows():
            images.append(row['data'])
            labels.append(row['label'])
            IDs.append(row['ID'])
        return images, labels, IDs

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


if __name__ == '__main__':
    rootpath = handle_path(r'E:\Verse')
    load_rawdata(rootpath)

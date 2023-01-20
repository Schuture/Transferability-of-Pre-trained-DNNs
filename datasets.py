import os
import random
from PIL import Image
from torch.utils.data import Dataset
        

class ImageNetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        ImageNet Dataset
        :param data_dir: str, path of the dataset
        :param transform: torch.transform，data pre-processing
        """
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform
        print('Data size: ', self.__len__())

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self, data_dir):
        data_info = list()
        class_names = sorted(os.listdir(data_dir))
        for i, class_name in enumerate(class_names):
            img_names = os.listdir(os.path.join(data_dir, class_name))
            img_names = list(filter(lambda x: x.endswith('.JPEG'), img_names))

            for j in range(len(img_names)):
                img_name = img_names[j]
                path_img = os.path.join(data_dir, class_name, img_name)
                label = i
                data_info.append((path_img, int(label)))

        return data_info


class ImageDataset(Dataset):
    def __init__(self, data_dir, is_train=True,
                 val_ratio=0.2, transform=None,
                 max_classes=9999, max_samples=1000000):
        """
        General Image Dataset
        :param data_dir: str, path of the dataset
        :param transform: torch.transform，data pre-processing
        """
        self.is_train = is_train
        self.val_ratio = val_ratio
        self.max_classes = max_classes
        self.max_samples = max_samples
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform
        print(data_dir, 'data size: ', self.__len__())

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label, path_img

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self, data_dir):
        data_info = list()
        for i, class_name in enumerate(os.listdir(data_dir)):
            img_names = os.listdir(os.path.join(data_dir, class_name))
            img_names = list(filter(lambda x: x.endswith('.jpg') or\
                                    x.endswith('.tif') or\
                                    x.endswith('.jpeg'), img_names))
            random.seed(1)
            random.shuffle(img_names)
            if self.is_train:
                img_names = img_names[int(self.val_ratio*len(img_names)):]
            else:
                img_names = img_names[:int(self.val_ratio*len(img_names))]

            # traverse images
            for j in range(len(img_names)):
                img_name = img_names[j]
                path_img = os.path.join(data_dir, class_name, img_name)
                label = i
                data_info.append((path_img, int(label)))
            
            if i+1 >= self.max_classes:
                break
        
        if len(data_info) > self.max_samples:
            random.shuffle(data_info)
            data_info = data_info[:self.max_samples]

        return data_info


if __name__ == '__main__':
    train_data = ImageNetDataset('/media/sribd/a3c90115-0a9e-4906-b659-f9555c821e81/数据集/ILSVRC2012_img_train')
    valid_data = ImageNetDataset('/media/sribd/a3c90115-0a9e-4906-b659-f9555c821e81/数据集/ILSVRC2012_img_val')













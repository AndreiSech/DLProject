import torch
import torchvision
import os
import pickle

class DataLoader:

    def __init__(self):
        self.BATCH_SIZE = 32
        self.mean = 0.5
        self.std = 0.2
        has_stats = False
        self.stats_path = '..\input\hymenoptera_data\\'
        self.stats_filename = os.path.join(self.stats_path, 'stats.pkl')

    def load_train_data(self, train_data_path: str, augmentation: bool) -> object:
        """
        Method used to extract data to train classification model
         train_data_path: path to folder with labeled pictures (train dataset)
         return DataLoader with batch_size=32
        """
        transformer = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                      torchvision.transforms.ToTensor()])

        train_data = get_data(train_data_path, transformer)
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True)
        self.mean, self.std = get_mean_and_std(train_loader, len(train_data.classes))
        stats = {
            'mean' : self.mean,
            'std' : self.std
        }
        with open(self.stats_filename, 'wb') as f:
            if not os.path.isdir(self.stats_path):
                os.makedirs(self.stats_path)
            pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)

        if (augmentation):
            transformer = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                                torchvision.transforms.AutoAugment(
                                                                torchvision.transforms.AutoAugmentPolicy.IMAGENET),
                                                                torchvision.transforms.ToTensor(),
                                                                torchvision.transforms.Normalize(self.mean, self.std)])
        else:
            transformer = torchvision.transforms.Compose([torchvision.transforms.Resize((244, 244)),
                                                            torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize(self.mean, self.std)])

        train_data = get_data(train_data_path, transformer)
        return torch.utils.data.DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True)


    def load_test_data(self, test_data_path: str) -> object:
        """
        Method used to extract data to validate metrics
         test_data_path: path to folder with labeled pictures (val dataset)
         return DataLoader with batch_size=32
        """
        try:
            with open(self.stats_filename, 'rb') as f:
                stats = pickle.load(f)
                self.mean = stats['mean']
                self.std = stats['std']
                self.has_stats = True
        except:
            print('file with statistics not found')

        #print("Mean: ", self.mean, ", std: ", self.std)

        transformer = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(self.mean, self.std)])

        test_data = get_data(test_data_path, transformer)
        return torch.utils.data.DataLoader(dataset=test_data, batch_size=self.BATCH_SIZE, shuffle=True)


def get_data(data_path, transformer):
    return torchvision.datasets.ImageFolder(data_path, transformer)


def get_mean_and_std(loader, n_classes):
    mean = std = total_images = 0
    for images, _ in loader:
        images = images.view(images.shape[0], images.shape[1], -1)
        mean += images.mean(n_classes).sum(0)
        std += images.std(n_classes).sum(0)
        total_images += images.shape[0]
    mean /= total_images
    std /= total_images
    return mean, std



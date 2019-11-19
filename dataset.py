from torch.utils import data
import glob
import cv2
from tqdm import tqdm

class Data(data.Dataset):
    def __init__(self, cat_images, dog_images):
        self.images = cat_images + dog_images
        self.labels = [0.] * len(cat_images) + [1.0] * len(dog_images)
        print("Number of images loaded: ", len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = self.images[index]
        y = self.labels[index]
        return X, y

class DataSplit():
    def __init__(self, dirpath, train, val, test):
        print("Preparing data...")
        cat_images = glob.glob(dirpath + "/Cat/*.jpg")[:1000]
        dog_images = glob.glob(dirpath + "/Dog/*.jpg")[:1000]

        # Filter out bad images
        print("\tFiltering corrupt images... ")
        cat_images = self.filter_corrupt_images(cat_images)
        dog_images = self.filter_corrupt_images(dog_images)

        # Compute train, val and test portions
        total = train + val + test
        train = train/total
        val = val/total
        test = test/total

        # compute number of images in train and val
        train_size = round(train * len(cat_images))
        val_size = round(val * len(cat_images))

        # Split into train, val and test
        print("\tSpliting dataset...")
        self.train_cats = cat_images[:train_size]
        self.train_dogs = dog_images[:train_size]
        self.val_cats = cat_images[train_size:train_size + val_size]
        self.val_dogs = dog_images[train_size:train_size + val_size]
        self.test_cats = cat_images[train_size + val_size:]
        self.test_dogs = dog_images[train_size + val_size:]

    def get_datasets(self):
        train_dataset = Data(self.train_cats, self.train_dogs)
        val_dataset = Data(self.val_cats, self.val_dogs)
        test_dataset = Data(self.test_cats, self.test_dogs)
        return train_dataset, val_dataset, test_dataset

    def filter_corrupt_images(self, paths):
        good_imgpaths = []
        for path in tqdm(paths):
            try:
                img = read_image(path)
                good_imgpaths.append(img)
            except:
                pass
        return good_imgpaths

def read_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, (80, 80))
    return img

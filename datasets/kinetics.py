###################################################################################################
#
# Copyright (C) 2022-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Classes and functions used to utilize the Kinetics dataset.
"""
import os
import ai8x
import numpy as np
import pickle
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
import yaml

class Kinetics(Dataset):
    """
    Kinetics400 Human Actions Dataset (400 action class)
    (https://deepmind.com/research/open-source/kinetics/).
    The image files are in RGB format and corresponding portrait matting files are in RGBA
    format where the alpha channel is 0 or 255 for background and portrait respectively.
    """
    def __init__(self, root, split, img_size, num_classes, fold_ratio, num_frames_model, num_frames_dataset,
                 transform, augmentation, blacklist_file=None, download=True, transformVideo=False):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.num_classes = num_classes
        self.fold_ratio = fold_ratio
        self.num_frames_model = num_frames_model
        self.num_frames_dataset = num_frames_dataset
        self.transform = transform
        self.augmentation = augmentation
        self.label_distribution = np.zeros(num_classes)
        self.blacklist_file = blacklist_file
        self.background_pickle_index = 0
        self.transformVideo = transformVideo
        self.datacounter = 0
        # Check split 
        if split not in ('test', 'train'):
            raise ValueError("Split name can only be set to 'test' or 'train'")   
        self.__load_dataset() 

    # Main dataloader function in the beginning
    def __load_dataset(self):

        # Load blacklist entries
        if self.blacklist_file is not None:
            with open(os.path.join(self.root, "kinetics400/blacklists", self.blacklist_file), 'r') as stream:
                blacklist_dict = yaml.load(stream, Loader=yaml.FullLoader)
                self.blacklist = [item for sublist in list(blacklist_dict.values()) for item in sublist]
                print(f'Blacklist loaded with {len(self.blacklist)} entries')
        else:
            self.blacklist = []

        # Load dataset samples
        self.dataset = []
        self.folder_name = f'kinetics400/processed_4class_fixed_{self.num_frames_dataset}frames_{self.img_size[0]+16}x{self.img_size[1]+16}'
        self.folder_path = os.path.join(self.root, self.folder_name, self.split)
        # todo_alican: bir yerde call os.path.join(self.root, self.__class__.__name__, 'raw') / 'processed'

        dir = sorted(os.listdir(self.folder_path))
        dir = [x for x in dir if x.endswith('.pkl')] # Take only pickle files
        print("I - ==========", self.split.upper(), " SET ==========")

        for pickle_filename in dir[0:self.num_classes-1]:                
            print(f'I - Loading file: {pickle_filename} in {self.folder_path}')
            pickle_filepath = os.path.join(self.folder_path, pickle_filename)
            with open(pickle_filepath, 'rb') as f:
                dataset = pickle.load(f)
                self.add_segment(dataset)
        self.data_wo_background = len(self.dataset) # Stores non-background sample count
    
    # Size of dataset
    def __len__(self):
        return len(self.dataset)

    # Item loader during epochs
    def __getitem__(self, index):

        print(self.datacounter)
        if self.datacounter%1000 == 0:
            print(self.datacounter)
            print('\n')

        self.datacounter += 1

        (imgs, lab, _) = self.dataset[index]

        start_ind = np.random.randint(low=0, high=(len(imgs)-self.num_frames_model+1)) # Randomly pick a frameModelNo long sequence
        images = imgs[start_ind:start_ind+self.num_frames_model]

        transforms_album = []
        if self.augmentation:
            transforms_album.append(A.RandomResizedCrop(height=images[0].shape[0]-16,
                                                        width=images[0].shape[0]-16,
                                                        scale=(0.5, 1.0),
                                                        ratio=(0.75, 1.3333333333333333),
                                                        p=1.0))
            transforms_album.append(A.HorizontalFlip(p=0.5))

            transform_album = A.Compose(transforms_album, additional_targets={
                'image0': 'image',
                'image1': 'image',
                'image2': 'image',
                'image3': 'image',
                'image4': 'image',
                'image5': 'image',
                'image6': 'image',
                'image7': 'image',
                'image8': 'image',
                'image9': 'image',
                'image10': 'image',
                'image11': 'image',
                'image12': 'image',
                'image13': 'image',
                'image14': 'image'
            })

            images_transformed = transform_album(image=images[0],
                                     image0=images[1],
                                     image1=images[2],
                                     image2=images[3],
                                     image3=images[4],
                                     image4=images[5],
                                     image5=images[6],
                                     image6=images[7],
                                     image7=images[8],
                                     image8=images[9],
                                     image9=images[10],
                                     image10=images[11],
                                     image11=images[12],
                                     image12=images[13],
                                     image13=images[14],
                                     image14=images[15]
                                     )

            for x in range(0, len(images_transformed)):
                if not x:
                    images[0] = images_transformed['image']
                else:
                    images[x] = images_transformed['image' + str(x - 1)]
        else:
            for x in range(0, len(images)):
                images[x] = images[x][8:248, 8:248, :]

        images_concat = []
        for x in range(len(images)-1):
            #diff = cvtColor(images[x], COLOR_RGB2GRAY) - cvtColor(images[x+1], COLOR_RGB2GRAY)
            images_concat.append(np.concatenate((images[x],images[x+1]),axis=2))
            #images_concat.append(np.concatenate((images[x], np.expand_dims(diff,axis=2)),axis=2))

        images = [self.fold_image(self.__normalize_image(img), self.fold_ratio) for img in images_concat] # Normalize and fold images
        
        if self.transform is not None:
            if self.transformVideo: # Used for other models (SlowFast, X3D)
                images = np.array(images).transpose((3,0,1,2))  
                images_final = self.transform(torch.FloatTensor(images))    
            else: # Apply given transform
                images_transformed = [self.transform(img) for img in images]
                images_list = [img.numpy() for img in images_transformed]
                images_final = torch.Tensor(np.array(images_list))
        else: # No transform
            images_list = images
            images_final = torch.Tensor(np.array(images_list).transpose((0,3,1,2)))
        return images_final, torch.tensor(lab, dtype=torch.long)

    @staticmethod
    def __normalize_image(image):
        return image / 255

    @staticmethod
    def fold_image(img, fold_ratio):
        """Folds high resolution H-W-3 image h-w-c such that H * W * 3 = h * w * c.
           These correspond to c/3 downsampled images of the original high resolution image."""
        if fold_ratio == 1:
            img_folded = img
        else:
            img_folded =np.empty((img.shape[0]//fold_ratio, img.shape[1]//fold_ratio, img.shape[2]*fold_ratio*fold_ratio), dtype=img.dtype)
            for i in range(fold_ratio):
                 for j in range(fold_ratio):
                    ch_idx = (i*fold_ratio + j) * img.shape[2]
                    img_folded[:, :, ch_idx:(ch_idx+img.shape[2])] = img[i::fold_ratio, j::fold_ratio, :]
        return img_folded
    
    # Append dataset with a single segment from each video (random starting point frame sequence)         
    def add_segment(self, dataset, blacklist_flag=True):
        for data in dataset:

            (imgs, lab, vidx) = data
            if vidx in self.blacklist and blacklist_flag:
                continue # Blacklist sample

            if len(imgs) > self.num_frames_dataset: # Check correct frame count
                print("I - Number of frames greater than dataset description, tossed video with #frames = ", len(imgs))
                continue
            
            imgs = imgs[1:-1] # Toss first and last frames
            l = len(imgs)
            if l<self.num_frames_model: # Check sufficient frame count
                print("I - Tossed video with insufficient frame number.")
                continue 
                
            self.dataset.append((imgs, lab, vidx))

            self.label_distribution[lab] += 1
            if lab in [1,2] and self.split == 'train': # Doubled classes for train set todo_alican: maybe get rid of double classes 
                self.dataset.append((imgs, lab, vidx))
                self.label_distribution[lab] += 1

    
    # Single pickle mode, background data loader (this is called at the beginning of each epoch)
    def load_next_background_pickle(self):    
        
        if len(self) > self.data_wo_background: # Delete current background data  
            del self.dataset[self.data_wo_background:]
            self.label_distribution[-1] = 0 

        folder_path = os.path.join(self.dir_path, self.folder_name, self.split, "new_background")
        dir = sorted(os.listdir(folder_path))
        dir = [x for x in dir if x.endswith('.pkl')] # Take only pkl files
        ind = self.background_pickle_index % len(dir) # Circular shift to next background pkl index
        self.background_pickle_index += 1
        pickle_filename = dir[ind]
        print(f'I - Loading file: {pickle_filename} in {folder_path}')
        pickle_filepath = os.path.join(folder_path, pickle_filename)
        with open(pickle_filepath, 'rb') as f:
            dataset = pickle.load(f)
            # Add data without blacklisting
            self.add_segment(dataset, blacklist_flag=False)
        print("I - New label distribution:", self.label_distribution)
        print('')

def kinetics_get_datasets(data, load_train=True, load_test=True, num_classes=4, img_size=(240,240),
        fold_ratio=4, num_frames_model=16, num_frames_dataset=50):
    """
    Load the folded 16 frame version of selected classes from the Kinetics 400 dataset

    The dataset is loaded from the archive file, so the file is required for this version.

    The dataset originally includes 400 action classes. A dataset is formed with 5 classes which
    includes 4 of the action classes and the a fraction of the rest of the dataset is used to
    form the last class, i.e class of the others. The dataset is split into training+validation and test sets.
    90:10 training+validation:test split is used by default.

    Data is augmented by random cropping of frames and flipping videos horizontally with 50% chance.
    """
    (data_dir, args) = data

    transform = transforms.Compose([transforms.ToTensor(), ai8x.normalize(args=args)])

    if num_classes == 4:
        classes = next((e for _, e in enumerate(datasets)
                        if len(e['output']) - 1 == num_classes))['output']
    else:
        raise ValueError(f'Unsupported num_classes {num_classes}')

    if load_train:
        train_dataset = Kinetics(root=data_dir, split='train', img_size=img_size, num_classes=len(classes), 
                                 fold_ratio=fold_ratio, num_frames_model=num_frames_model, num_frames_dataset=num_frames_dataset,
                                 transform=transform, augmentation=True, blacklist_file="blacklist100.yaml", download=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = Kinetics(root=data_dir, split='test', img_size=img_size, num_classes=len(classes), 
                                fold_ratio=fold_ratio, num_frames_model=num_frames_model, num_frames_dataset=num_frames_dataset,
                                transform=transform, augmentation=False)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset

datasets = [
    {
        'name': 'Kinetics400',
        'input': (6, 240, 240),
        'output': ('pull up', 'push up', 'situp', 'squat', 'other'),
        'weight': (0.2, 0.25, 0.2, 0.25, 0.1),
        'loader': kinetics_get_datasets,
    },
]

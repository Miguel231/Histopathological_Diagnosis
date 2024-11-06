import os
import random
from PIL import Image

class TripletDataset:
    def __init__(self, df, path, train=True, transform=None):
        self.data_csv = df
        self.is_train = train
        self.transform = transform
        self.path = path
        
        if self.is_train:
            self.patients = df['Pat_ID'].values  # Use Pat_ID to identify patients
            self.images = df['Window_ID'].values  # Use Windows_ID to identify image patches
            self.labels = df['Presence'].values  # Use presence for labels
            self.index = df.index.values 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        patient_id = self.patients[item]
        window_id = str(self.images[item]).zfill(5)
        anchor_image_path = os.path.join(self.path, f"{patient_id}_{window_id}.png")
        
        if not os.path.exists(anchor_image_path):
            print(f"Warning: Anchor image file not found: {anchor_image_path}")
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        anchor_img = Image.open(anchor_image_path).convert('RGB')
        anchor_label = self.labels[item]
        
        positive_img, positive_image_path = self._get_positive_sample(item, anchor_label)
        negative_img, negative_image_path = self._get_negative_sample(item, anchor_label)

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img, anchor_label

    def _get_positive_sample(self, item, anchor_label):
        positive_list = self.index[self.index!=item][self.labels[self.index!=item]==anchor_label]
        
        # Retry until a valid positive item is found
        while True:
            if len(positive_list) == 0:
                raise ValueError(f"No positive sample found for anchor label {anchor_label} at index {item}")

            positive_item = random.choice(positive_list)
            
            if positive_item >= len(self.patients):
                print(f"Warning: Positive index {positive_item} out of bounds. Retrying...")
                positive_list = positive_list[positive_list != positive_item]
                continue  # Retry with a new sample if out of bounds

            positive_patient_id = self.patients[positive_item]
            positive_window_id = str(self.images[positive_item]).zfill(5)
            positive_image_path = os.path.join(self.path, f"{positive_patient_id}_{positive_window_id}.png")

            if not os.path.exists(positive_image_path):
                print(f"Warning: Positive image file not found: {positive_image_path}. Retrying...")
                positive_list = positive_list[positive_list != positive_item]
                continue

            return Image.open(positive_image_path).convert('RGB'), positive_image_path

    def _get_negative_sample(self, item, anchor_label):
        negative_list = self.index[self.index!=item][self.labels[self.index!=item]!=anchor_label]
        
        # Retry until a valid negative item is found
        while True:
            if len(negative_list) == 0:
                raise ValueError(f"No negative sample found for anchor label {anchor_label} at index {item}")

            negative_item = random.choice(negative_list)
            
            if negative_item >= len(self.patients):
                print(f"Warning: Negative index {negative_item} out of bounds. Retrying...")
                negative_list = negative_list[negative_list != negative_item]
                continue  # Retry with a new sample if out of bounds

            negative_patient_id = self.patients[negative_item]
            negative_window_id = str(self.images[negative_item]).zfill(5)
            negative_image_path = os.path.join(self.path, f"{negative_patient_id}_{negative_window_id}.png")

            if not os.path.exists(negative_image_path):
                print(f"Warning: Negative image file not found: {negative_image_path}. Retrying...")
                negative_list = negative_list[negative_list != negative_item]
                continue

            return Image.open(negative_image_path).convert('RGB'), negative_image_path

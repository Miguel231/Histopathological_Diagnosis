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
        # Construct the path for the anchor image using both Pat_ID and Windows_ID
        patient_id = self.patients[item]
        window_id = str(self.images[item]).zfill(5)
        anchor_image_path = os.path.join(self.path, f"{patient_id}_{window_id}.png")  # Adjust extension if needed
        
        # Check if the anchor image exists
        if not os.path.exists(anchor_image_path):
            print(f"Warning: Anchor image file not found: {anchor_image_path}")
            raise FileNotFoundError(f"Image file not found: {anchor_image_path}")

        # Load the anchor image
        anchor_img = Image.open(anchor_image_path).convert('RGB')
        
        if self.is_train:
            anchor_label = self.labels[item]
            print("anchor", anchor_label)
            # Find a positive example with the same label, excluding the anchor
            positive_list = self.index[(self.index != item) & (self.labels == anchor_label)]
            positive_item = random.choice(positive_list)
            positive_patient_id = self.patients[positive_item]
            positive_window_id = self.images[positive_item]
            positive_image_path = os.path.join(self.path, f"{positive_patient_id}_{positive_window_id}")

            # Check if the positive image exists
            if not os.path.exists(positive_image_path):
                print(f"Warning: Positive image file not found: {positive_image_path}")
                raise FileNotFoundError(f"Image file not found: {positive_image_path}")

            positive_img = Image.open(positive_image_path).convert('RGB')

            # Find a negative example with a different label
            negative_list = self.index[(self.index != item) & (self.labels != anchor_label)]
            negative_item = random.choice(negative_list)
            negative_patient_id = self.patients[negative_item]
            negative_window_id = self.images[negative_item]
            negative_image_path = os.path.join(self.path, f"{negative_patient_id}_{negative_window_id}")  # Ensure consistent extension

            # Check if the negative image exists
            if not os.path.exists(negative_image_path):
                print(f"Warning: Negative image file not found: {negative_image_path}")
                raise FileNotFoundError(f"Image file not found: {negative_image_path}")

            negative_img = Image.open(negative_image_path).convert('RGB')

            # Apply transformations if provided
            if self.transform is not None:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            return anchor_img, positive_img, negative_img, anchor_label

        # If not in training mode, only return the anchor image and its label
        if self.transform is not None:
            anchor_img = self.transform(anchor_img)

        return anchor_img, anchor_label

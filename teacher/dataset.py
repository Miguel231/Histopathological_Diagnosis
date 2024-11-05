from PIL import Image
import random

class TripletDataset():
    def __init__(self, df, path, train=True, transform=None):
        self.data_csv = df
        self.is_train = train
        self.transform = transform
        self.path = path
        if self.is_train:
            self.images = df.iloc[:, 1].values  # Assumes image filenames are in the second column
            self.labels = df.iloc[:, 2].values  # Assumes labels are in the third column
            self.index = df.index.values 
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):
        # Construct the path for the anchor image
        anchor_image_name = self.images[item]
        anchor_image_path = f"{self.path}/{anchor_image_name}"
        
        # Load the anchor image
        anchor_img = Image.open(anchor_image_path).convert('RGB')
        
        if self.is_train:
            # Get the label for the anchor image
            anchor_label = self.labels[item]
            
            # Find a positive example with the same label, excluding the anchor
            positive_list = self.index[(self.index != item) & (self.labels == anchor_label)]
            positive_item = random.choice(positive_list)
            positive_image_name = self.images[positive_item]
            positive_image_path = f"{self.path}/{positive_image_name}"
            positive_img = Image.open(positive_image_path).convert('RGB')
            
            # Find a negative example with a different label
            negative_list = self.index[(self.index != item) & (self.labels != anchor_label)]
            negative_item = random.choice(negative_list)
            negative_image_name = self.images[negative_item]
            negative_image_path = f"{self.path}/{negative_image_name}"
            negative_img = Image.open(negative_image_path).convert('RGB')
            
            # Apply transformations if provided
            if self.transform is not None:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)
                
        # Return the triplet (anchor, positive, negative) and the anchor label
        return anchor_img, positive_img, negative_img, anchor_label

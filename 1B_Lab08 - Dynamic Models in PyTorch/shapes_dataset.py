import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os

class ShapesDataset(Dataset):
    def __init__(self, stage = 'easy', num_instances = 10000, image_size = 64, random_seed = 42):
        super().__init__()
        np.random.seed(random_seed)
        self.stage = stage
        self.num_instances = num_instances
        self.image_size = image_size
        self.shapes = ['circle', 'square', 'triangle']

    def __len__(self):
        return self.num_instances

    def __getitem__(self, idx):
        # start with an image with a white background
        img = np.ones((self.image_size, self.image_size), dtype = np.uint8) * 255
        label = np.random.randint(0, len(self.shapes))
        shape_type = self.shapes[label]

        if self.stage == 'easy':
            img = self.draw_shape(img, shape_type, centered = True, big = True)
        elif self.stage == 'medium':
            # draw one small, random_position shape
            img = self.draw_shape(img, shape_type, 
                                  centered = False, big = False, rotate = True, scale = True)
        else: # 'hard'
            # draw multiple overlapping shapes
            num_shapes = np.random.randint(2, 5)
            for _ in range(num_shapes):
                img = self.draw_shape(img, shape_type, 
                                      centered = False, big = False, rotate = True, scale = True)

        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)
        label = torch.tensor(label)
        return img, label

    def draw_shape(self, img, shape_type, centered, big, rotate=False, scale=False):
        upscale_factor = 4 
        h, w = img.shape
        h_hr, w_hr = h * upscale_factor, w * upscale_factor 

        mask_hr = np.zeros((h_hr, w_hr), dtype=np.uint8)

        if big:
            base_size = np.random.randint(
                int(0.3 * self.image_size),
                int(0.4 * self.image_size))
        else:
            base_size = np.random.randint(
                int(0.2 * self.image_size),
                int(0.3 * self.image_size))
        
        if scale:
            scale_factor = np.random.uniform(0.5, 1.2)
            size = int(base_size * scale_factor * upscale_factor)
        else:
            size = base_size * upscale_factor

        center_x = w_hr // 2 if centered else np.random.randint(size, w_hr - size)
        center_y = h_hr // 2 if centered else np.random.randint(size, h_hr - size)

        if shape_type == 'circle':
            cv2.circle(mask_hr, (center_x, center_y), size, (255,), thickness=-1, lineType=cv2.LINE_AA)
        elif shape_type == 'square':
            top_left = (center_x - size, center_y - size)
            bottom_right = (center_x + size, center_y + size)
            cv2.rectangle(mask_hr, top_left, bottom_right, (255,), thickness=-1, lineType=cv2.LINE_AA)
        elif shape_type == 'triangle':
            pts = np.array([
                [center_x, center_y - size],
                [center_x - size, center_y + size],
                [center_x + size, center_y + size]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask_hr, [pts], (255,))

        if rotate:
            angle = np.random.uniform(0, 360)
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            mask_hr = cv2.warpAffine(mask_hr, M, (w_hr, h_hr), flags=cv2.INTER_LINEAR, borderValue=0)

        mask = cv2.resize(mask_hr, (w, h), interpolation=cv2.INTER_AREA)
        img = np.minimum(img, 255 - mask)

        return img

def show_images(easy_dataset, medium_dataset, hard_dataset, images_per_stage = 5):
    stages = {
        'easy' : easy_dataset,
        'medium' : medium_dataset,
        'hard' : hard_dataset
        }

    shape_labels = ['circle', 'square', 'triangle']
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    fig, axes = plt.subplots(len(stages), images_per_stage,
                             figsize = (images_per_stage * 2, len(stages) * 2)) 

    for row_idx, (stage_name, dataset) in enumerate(stages.items()):
        for col_idx in range(images_per_stage):
            img, label = dataset[np.random.randint(len(dataset))]
            img = img.squeeze(0).numpy() 
            ax = axes[row_idx, col_idx]
            ax.imshow(img, cmap = 'gray')
            ax.grid(True) 
            ax.axis('on') 
            ax.set_title(f'{stage_name}\n{shape_labels[label]}')
    plt.subplots_adjust(wspace=0.2, hspace=0.8)
    plt.tight_layout()


    file_path = os.path.join(plots_dir, 'dataset_samples.png')
    plt.savefig(file_path)
    plt.close(fig)
    print(f"Dataset sample images saved to {file_path}")














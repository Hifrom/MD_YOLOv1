import torch
from model import YoloTiny
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import models
import torch.nn as nn
from utils import (non_max_suppression,
                   mean_average_precision,
                   intersection_over_union,
                   get_bboxes,
                   plot_image,
                   plot_image2,
                   load_checkpoint,
                   cellboxes_to_boxes)
from dataset import Dataset
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
import numpy as np
DEVICE = 'cuda'


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

transform = Compose([transforms.ToTensor(), transforms.Resize(size=224), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def main():
    path_to_model = 'epoch_num_450.pth'
    model = models.mobilenet_v2(pretrained=True, progress=True).to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(nn.Linear(1280, 1000), nn.Dropout(p=0.5), nn.LeakyReLU(0.1), nn.Linear(1000, 539)).to(DEVICE)
    optimizer = None
    load_checkpoint(torch.load(path_to_model), model, optimizer)
    model.eval()
    #test_dataset = Dataset("data/test.csv", transform=transform, img_dir='data/images', label_dir='data/labels')
    #test_loader = DataLoader(dataset=test_dataset, batch_size=16, num_workers=2, pin_memory=True, shuffle=True, drop_last=False)

    #pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4)
    #mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format='midpoint', )
    #print(f"Train mAP: {mean_avg_prec}")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Warmup Camera
    print('Warmup Camera')
    for i in tqdm(range(5)):
        ret, frame = cap.read()
    # while(True):
        '''
        
        _, frame = cap.read()
        frame = cv2.resize(frame, (416, 416))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_tensor = torch.from_numpy(frame).float()
        frame_tensor = frame_tensor.permute(2, 0, 1)
        frame_tensor = torch.unsqueeze(frame_tensor, 0).to('cuda')
        
        img = cv2.imread('170.jpg')
        print('Picture before: {0}'.format(img))
        frame_tensor = torch.from_numpy(img).float()
        frame_tensor = frame_tensor.permute(2, 0, 1)
        frame_tensor = torch.unsqueeze(frame_tensor, 0).to('cuda')
        print('frame_tensor: {0}'.format(frame_tensor))
        for idx in range(1):
            bboxes = cellboxes_to_boxes(model.forward(frame_tensor))
            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            plot_image2(frame_tensor[idx].permute(1, 2, 0).to("cpu"), bboxes)
            plot_image(img, bboxes)
          
    # cap.release()
    # cv2.destroyAllWindows()
    
    for x, y in test_loader:
        x = x.to('cuda')
        for idx in range(8):
            bboxes = cellboxes_to_boxes(model(x))
            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            print(bboxes)
            plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
    '''
    k = 0
    while True:
        _, img = cap.read()
        img = cv2.resize(img, (224, 224))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_tensor = torch.from_numpy(img).float()
        frame_tensor = frame_tensor.permute(2, 0, 1) / 255.0
        frame_tensor = torch.unsqueeze(frame_tensor, 0).to(DEVICE)
        x = frame_tensor.to(DEVICE)
        for idx in range(1):
            bboxes = cellboxes_to_boxes(model.forward(x))
            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.4, threshold=0.5, box_format="midpoint")
            plot_image(img, bboxes, k)
            # plot_image2(x[idx].permute(1,2,0).to("cpu"), bboxes)
        k += 1

if __name__ == '__main__':
    main()
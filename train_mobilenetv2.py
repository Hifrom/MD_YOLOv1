import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader
from model import YoloTiny
from dataset import Dataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

seed = 101
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cuda"
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = 'epoch_num_30.pth' # "overfit.pth.tar"
IMG_DIR_TRAIN = "data/images"
LABEL_DIR_TRAIN = "data/labels"
IMG_DIR_VALID = 'data/valid_images'
LABEL_DIR_VALID = 'data/valid_labels'

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

transform = Compose([transforms.ToTensor(), transforms.Resize(size=224), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model.forward(x)
        loss = loss_fn.forward(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    log = open('log.txt', 'a')
    string = 'Mean Loss: ' + str(sum(mean_loss)/len(mean_loss)) + '\n'
    log.write(string)
    log.close()

def main():
    model = models.mobilenet_v2(pretrained=True, progress=True).to(DEVICE)
    for param in model.parameters():
        param.requires_grad = True
    model.classifier = nn.Sequential(nn.Linear(1280, 1000), nn.Dropout(p=0.5), nn.LeakyReLU(0.1), nn.Linear(1000, 539)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
        model.train()
    train_dataset = Dataset("data/train.csv", transform=transform, img_dir=IMG_DIR_TRAIN, label_dir=LABEL_DIR_TRAIN)
    valid_dataset = Dataset("data/valid.csv", transform=transform, img_dir=IMG_DIR_VALID, label_dir=LABEL_DIR_VALID)
    train2_dataset = Dataset("data/train2.csv", transform=transform, img_dir='data/images', label_dir='data/labels')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, shuffle=True,
                              drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, shuffle=True,
                              drop_last=True)
    train2_loader = DataLoader(dataset=train2_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, shuffle=True,
                              drop_last=True)


    for epoch in range(31, EPOCHS):
        print('************ Epoch Number: {0} ************'.format(int(epoch)))
        # if epoch == 10:
        #     for x, y in test_loader:
        #        x = x.to(DEVICE)
        #        for idx in range(8):
        #            bboxes = cellboxes_to_boxes(model(x))
        #            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #            plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
            #
               # import sys
               # sys.exit()

        pred_boxes, target_boxes = get_bboxes(train2_loader, model, iou_threshold=0.5, threshold=0.4)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        one = mean_avg_prec
        print(f"Train mAP: {mean_avg_prec}")
        pred_boxes, target_boxes = get_bboxes(valid_loader, model, iou_threshold=0.5, threshold=0.4)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        two = mean_avg_prec
        print(f"Valid mAP: {mean_avg_prec}")

        # if mean_avg_prec > 0.9:
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'epoch_num_' + str(epoch) + '.pth')
           # checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
           # save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE + str(epoch) + '.pth')
           # import time
           # time.sleep(10)
        log = open('log.txt', 'a')
        string = '********************** Epoch №: ' + str(epoch) + ' **********************\n'
        log.write(string)
        string = 'Train mAP: ' + str(one) + '\n'
        log.write(string)
        string = 'Valid mAP: ' + str(two) + '\n'
        log.write(string)
        log.close()
        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()

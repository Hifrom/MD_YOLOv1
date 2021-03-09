import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn

# Mirror Images and save;
# Also txt files are saved with mirror augmented;
'''
path_razmetka = 'D:/Study/_4_sem/_kursach/images_webcam2/norm_razmetka/'
path_razmetka_save = 'D:/Study/_4_sem/_kursach/all_webcam/'
k = 3266

# Mirror Image
for i in range(1612, 1956):
    _data_to_write = []
    image = cv2.imread(path_razmetka + str(i) + '.jpg')
    mirror_image = cv2.flip(image, 1)
    cv2.imwrite(path_razmetka_save + str(k) + '.jpg', mirror_image)
    f = open(path_razmetka + str(i) + '.txt', 'r')
    for line in f:
        data = line
        data_list = [x for x in data.split()]
        # print(data_list)
        data_list[1] = str(1.0 - float(data_list[1]))
        data_list[2] = str(float(data_list[2]))
        # print(data_list)
        data_to_write = ''
        for dat in data_list:
            data_to_write += dat + ' '
        data_to_write = data_to_write[0:-1]
        _data_to_write.append(data_to_write + '\n')
    f.close()
    f = open(path_razmetka_save + str(k) + '.txt', 'w')
    for string in _data_to_write:
        f.write(string)
    f.close()
    print(i)
    k += 1
'''
'''
# Recount Images
path = 'D:/Study/_4_sem/_kursach/images_webcam2/norm/'
path_to_save = 'D:/Study/_4_sem/_kursach/images_webcam2/norm/'
k = 1612
for i in range(1, 345):
    img = cv2.imread(path + '0 (' + str(i) + ').jpg')
    cv2.imwrite(path_to_save + str(k) + '.jpg', img)
    k += 1
'''
'''
# Rename txt
path = 'D:/Study/_4_sem/_kursach/all/'
k = 1957
for i in range(1, 1310):
    os.rename(path + str(i) + '.txt', path + str(k) + '.txt')
    k += 1
'''
'''
# Rename .jpg
path = 'D:/Study/_4_sem/_kursach/images_webcam2/valid/'
k = 186
for i in range(234, 312):
    os.rename(path + str(i) + '_test.jpg', path + str(k) + '.jpg')
    k += 1
'''
'''
# Video Capture and Images for Inference
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (416, 416))
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_tensor = torch.from_numpy(frame)
    frame_tensor = frame_tensor.permute(2,0,1)
    frame_tensor = torch.unsqueeze(frame_tensor, 0)
    print(frame_tensor.shape)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
'''

'''
#Resize one image
img = cv2.imread('5.jpg')
img_res = cv2.resize(img, (224, 224))
cv2.imwrite('5_res.jpg', img_res)
'''


#Draw Graphs of Training
path_txt = 'log.txt'
file = open(path_txt)
max_map = -1
min_error = 99999999999
num_epoch = 0
epoch = []
train_map = []
valid_map = []
mean_loss = []
for line in file:
    if line[0] == '*':
        epoch.append(int(num_epoch))
        num_epoch += 1
    if line[0] == 'T':
        train_map.append(float(line[18:-2]))
    if line[0] == 'V':
        valid_map.append(float(line[18:-2]))
        if (float(line[18:-2])) > max_map:
            max_map = float(float(line[18:-2]))
            epoch_map = num_epoch
    if line[0] == 'M':
        mean_loss.append(float(line[11:-1]))
        if float(line[11:-1]) < min_error:
            min_error = float(line[11:-1])
            epoch_error = num_epoch
file.close()
print(epoch)
print(train_map)
print(valid_map)
print(mean_loss)
print('Max Test mAP: {0} on Epoch № {1}'.format(max_map, epoch_map))
print('Min Error: {0} on Epoch № {1}'.format(min_error, epoch_error))
leg1 = ['Train mAP', 'Test mAP']
leg2 = ['Mean Loss']

plt.figure(1)
plt.subplot(211)
plt.plot(epoch, train_map)
plt.plot(epoch, valid_map)
plt.legend(leg1)
plt.ylabel('mAP')

plt.subplot(212)
plt.plot(epoch, mean_loss)
plt.legend(leg2)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()


# Transfer Learning
    # Vgg11
'''
vgg11 = models.vgg11(pretrained=True,progress=True) # Скачали обученную модель
model = models.vgg11(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Sequential(nn.Linear(4096, 539))
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')
'''

'''
    # MobileNetV2
model = models.mobilenet_v2(pretrained=True, progress=True)
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(nn.Linear(1280, 1000), nn.Dropout(p=0.5), nn.LeakyReLU(0.1), nn.Linear(1000, 539))
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
   p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')
'''

import cv2

'''
path_open = 'D:/Study/_4_sem/_kursach/images/'
path_save = 'D:/Study/_4_sem/_kursach/images_mirror/'
for i in range(0, 675):
    if i == 201 or i == 395 or i == 496:
        continue
    k = i + 675
    print('image: {0}'.format(i))
    image = cv2.imread(path_open + str(i) + '.jpg')
    mirror_image = cv2.flip(image, 1)
    cv2.imwrite(path_save + str(k) + '.jpg', mirror_image)
'''

'''
# Image Numeric
path_open = 'D:/Study/_4_sem/_kursach/images/'
path_save = 'D:/Study/_4_sem/_kursach/images_numeric/'
k = 0
for i in range(0, 675):
    if i in [199, 200, 201, 390, 391, 392, 393, 394, 395, 496, 502, 503, 504, 507, 508, 509, 510, 511, 512, 561]:
        continue
    print('image: {0}'.format(i))
    image = cv2.imread(path_open + str(i) + '.jpg')
    cv2.imwrite(path_save + str(k) + '.jpg', image)
    k += 1
'''

# Mirror Image Numeric
path_open = 'D:/Study/_4_sem/_kursach/images_numeric/'
path_save = 'D:/Study/_4_sem/_kursach/images_mirror/'
for i in range(0, 655):
    k = i + 655
    print('image: {0}'.format(i))
    image = cv2.imread(path_open + str(i) + '.jpg')
    mirror_image = cv2.flip(image, 1)
    cv2.imwrite(path_save + str(k) + '.jpg', mirror_image)
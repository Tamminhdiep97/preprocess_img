import os
import cv2
from pathlib import Path


def read_image(dir):
    folder_image = Path(dir)
    images = []
    for file in folder_image.iterdir():
        if file.name != '.DS_Store':
            print(file)
            img = cv2.imread(str(file), 1)
            images.append(img)
    
    return images






def fusion_Image(dir):
    images = read_image(dir)
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)

    mergeMertens = cv2.createMergeMertens()
    exposureFusion = mergeMertens.process(images)

    print(type(exposureFusion))
    cv2.imshow('Fusion', exposureFusion)
    while True:
        k = cv2.waitKey(33)
        if k == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    fusion_Image('change_light2')

import numpy as np
import cv2
import os
import csv
import sys

TXT_EXT = '.txt'

classes = ['articulated_truck', 'bicycle', 'bus', 'car',
           'motorcycle', 'motorized_vehicle', 'non-motorized_vehicle',
           'pedestrian', 'pickup_truck',
           'single_unit_truck', 'work_van']


def BndBox2YoloLine(bbox, height, width):
    pts = bbox['bbox']

    xmin = int(pts[0])
    ymin = int(pts[1])
    xmax = int(pts[2])
    ymax = int(pts[3])

    xcen = float((xmin + xmax)) / 2 / width
    ycen = float((ymin + ymax)) / 2 / height

    w = float((xmax - xmin)) / width
    h = float((ymax - ymin)) / height

    label = bbox['class']
    if label not in classes:
        classes.append(label)

    classIndex = classes.index(label)

    return classIndex, xcen, ycen, w, h


def main():
    if len(sys.argv) < 3:
        print("\nUsage : \n\t python3 mio-dataset-converter.py PATH CSV_FILE_NAME\n")
        print("\t PATH : path to the folder containing the training images ")
        print("\t CSV_FILE_NAME : csv file containing the bounding boxes (typically gt_train.csv) \n")
        return

    train_label = {}
    with open(sys.argv[2], 'r') as f:
        reader = csv.reader(f, delimiter=',')
        last_img_row = '00000000'
        for row in reader:
            img_name = row[0]
            if last_img_row != img_name:
                path = os.path.join(sys.argv[1], str(last_img_row) + '.jpg')
                img = cv2.imread(path)
                out_file = open("converted/" + str(last_img_row) + TXT_EXT, 'w', encoding="utf-8")
                height, width, _ = img.shape
                for bbox in train_label[last_img_row]:
                    classIndex, xcen, ycen, w, h = BndBox2YoloLine(bbox, height, width)

                    out_file.write("%d %.6f %.6f %.6f %.6f\n" % (classIndex, xcen, ycen, w, h))
                print(f'{last_img_row}.txt Convertido!')
                out_file.close()
                last_img_row = img_name
                train_label.clear()

            bbox = {'class': row[1], 'bbox': np.array(row[2:]).astype('int32')}

            if img_name in train_label:
                train_label[img_name].append(bbox)
            else:
                train_label[img_name] = [bbox]
        print(f'Conversao finalizada!')


if __name__ == '__main__':
    main()

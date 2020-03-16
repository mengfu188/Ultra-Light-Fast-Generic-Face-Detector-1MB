import pickle
import numpy as np
import cv2
from pathlib import Path


class BrainDataset:
    """
    self.img_infos = [{id, filename, width, height}]
    self.img_ids = [id]
    该数据集给予[OID_ToolKit](https://github.com/EscVM/OIDv4_ToolKit)项目
    直接读取图片,标签为配置中指定的类,bbox为(left, top, right, bottom)
    """

    def __init__(self, file, transform=None, target_transform=None, filter_size=0, **kwargs):
        # self.class_description = kwargs.get('class_description')
        self.classes = self.class_names = kwargs.get('classes') or ('BACKGROUND', 'Human head')
        # 要使用全部数据还是部分数据[0, 1]
        self.persentage = kwargs.get('persentage') or 1
        self.transform = transform
        self.target_transform = target_transform
        self.filter_size = filter_size

        with open(file, 'rb') as f:
            data = pickle.load(f)  # type is dict
        # 去掉负类
        data = data.values()  # each elem has img, label(pos or neg) bbox(x1, y1, w, h
        data = [x for x in data if x[1] != 0]

        # data = [x[0][:, ] for x in data]
        new_data = []
        for x in data:
            image, label, bboxes = x
            bboxes[:, 2:] = bboxes[:, 0:2]+bboxes[:, 2:]
            new_data.append([image, bboxes, 1])

        self.data = new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ann = self.data[index]

        image, boxes, labels = ann

        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

def show_and_draw(data):
    image, bbox, label = data
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    for box in bbox:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
    cv2.imshow('', image)
    cv2.waitKey()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    # dataset = BrainDataset('/home/cmf/datasets/brainwash/data_list_brainwash_test.pkl')
    dataset = BrainDataset('/home/cmf/datasets/brainwash/data_list_brainwash_train.pkl')

    for i in range(len(dataset)):
        show_and_draw(dataset[i])
        image, boxes, labels = dataset[i]
        # cv2.imshow('', image)
        # cv2.waitKey()
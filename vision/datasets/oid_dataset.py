import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import time
from pathlib import Path
import cv2


class OIDDataset:
    """
    self.img_infos = [{id, filename, width, height}]
    self.img_ids = [id]
    该数据集给予[OID_ToolKit](https://github.com/EscVM/OIDv4_ToolKit)项目
    直接读取图片,标签为配置中指定的类,bbox为(left, top, right, bottom)
    """

    def __init__(self, root, transform=None, target_transform=None, type='train', filter_size=0, **kwargs):
        # self.class_description = kwargs.get('class_description')
        self.classes = self.class_names = kwargs.get('classes') or ('BACKGROUND', 'Human head')
        # 要使用全部数据还是部分数据[0, 1]
        self.persentage = kwargs.get('persentage') or 1
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.filter_size = filter_size
        # classes = getattr(kwargs, 'classes')
        # delattr(kwargs, 'classes')
        self.class_description = str(self.root / 'csv_folder' / 'class-descriptions-boxable.csv')
        self.annotations_file = str(self.root / 'csv_folder' / f'{type}-annotations-bbox.csv')
        self.img_prefix = str(self.root / 'Dataset' /type / 'Human head')

        self.img_infos = self.load_annotations()

        pass

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, index):
        ann = self.get_ann_info(index)

        image = self._read_image(ann['filename'])
        boxes = ann['ann']['bboxes']
        labels = ann['ann']['labels']

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def _read_image(self, filename):
        image = cv2.imread(os.path.join(self.img_prefix, str(filename)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_annotations(self):
        """
        读取self.img_prefix文件夹下的Label中的文件，以该文件夹为基础读取图片
        返回值给self.img_infos， self.img_info需要{filename, width, height}
        :return:
        """
        # ann = pd.read_csv(ann_file)
        code_class_df = pd.read_csv(self.class_description, header=None, names=['code', 'class'])
        # print(code_class_df.head())

        code_class = dict(zip(code_class_df['code'], code_class_df['class']))
        class_code = dict(zip(code_class_df['class'], code_class_df['code']))

        ann_csv = pd.read_csv(self.annotations_file)
        print(ann_csv.head())

        def add_class(df, code_class):
            df['class'] = [code_class[x] for x in df['LabelName'].to_list()]

        add_class(ann_csv, code_class)

        print(ann_csv.head())

        obj_classes = self.classes

        obj_imgs = ann_csv[ann_csv['class'].isin(obj_classes)]

        filenames = [x for x in os.listdir(self.img_prefix) if x.endswith('jpg')]
        filenames = filenames[0:int(len(filenames) * self.persentage)]
        img_ids = [x.split('.')[0] for x in filenames]

        print('folder {} path have {} number imgs'.format(self.img_prefix, len(obj_imgs)))

        # 数据集中的imgids
        # img_ids = obj_imgs['ImageID']
        # img_ids =
        # 若在Label文件中有的标识，则应该存在对应的图片
        # filenames = ['{}.jpg'.format(x) for x in img_ids]

        # 确定width， height
        widths, heights = [], []
        for file in tqdm(filenames):
            file_path = os.path.join(self.img_prefix, file)
            img = Image.open(file_path)
            widths.append(img.width)
            heights.append(img.height)

            img.close()

        # label格式：[class_name xmin ymin xmax ymax\n...]
        # Orange 0.0 0.0 793.7986559999999 765.0
        # 一次性读取ann，若内存不够可讲下面的模块搬到`get_ann_info`方法中
        bboxes = []
        labels = []

        classes_id = dict([(x, i) for i, x in enumerate(self.classes)])

        groups = obj_imgs.groupby('ImageID')

        print(classes_id)

        # sub_obj_imgs

        for i, img_id in enumerate(tqdm(img_ids)):
            label = []
            bbox = []
            width = widths[i]
            height = heights[i]
            # start = time.time()
            # sub_anns = obj_imgs[obj_imgs['ImageID'] == img_id]
            sub_anns = groups.get_group(img_id)
            # sub_anns = obj_imgs[(obj_imgs['ImageID'] == img_id) & (obj_imgs['class'].isin(self.CLASSES))]
            # print(time.time() - start)

            # bbox = sub_anns[['XMin', 'XMax', 'YMin', 'YMax']].values.tolist()
            # label = [classes_id[x] for x in sub_anns['class'].values.tolist()]

            for sub_ann in sub_anns.iterrows():
                content = sub_ann[1]
                c = content['class']
                if c not in self.classes:
                    continue
                left = width * float(content['XMin'])
                top = height * float(content['YMin'])
                right = width * float(content['XMax'])
                bottom = height * float(content['YMax'])

                # 过滤掉小方框
                if right - left < self.filter_size or bottom - top < self.filter_size:
                    print(f'oid filter {img_id} bbox is {left} {top} {right} {bottom}')
                    continue

                bbox.append([left, top, right, bottom])
                label.append(classes_id[content['class']])

            if len(bbox) == 0:
                print(f'oid filter {img_id} img because bbox length is 0')
                continue

            bboxes.append(np.asarray(bbox).astype(np.float32))
            labels.append(np.asarray(label).astype(np.int64))

        img_infos = [{
            'filename': filename,
            'width': width,
            'height': height,
            'ann': {
                'bboxes': bbox,
                'labels': label,
                'bboxes_ignore': np.array([], dtype=np.float32).reshape(-1, 4),
                'labels_ignore': np.array([], dtype=np.int64)
            }
        } for filename, width, height, bbox, label
            in zip(filenames, widths, heights, bboxes, labels)]

        return img_infos

    def get_ann_info(self, idx):
        """
        获得对应idx的ann
        返回{
            'bboxes': <np.ndarray, float32> (n, 4),
            'labels': <np.ndarray, int64> (n, ),
            'bboxes_ignore': <np.ndarray, float32> (k, 4),
            'labels_ignore': <np.ndarray, int64> (k, ) (optional field)
        }
        :param idx:
        :return:
        """
        return self.img_infos[idx]


if __name__ == '__main__':
    dataset = OIDDataset('/home/cmf/datasets/open-image/OID', type='validation')

    for i in dataset:
        data = i

        print(data)
        break

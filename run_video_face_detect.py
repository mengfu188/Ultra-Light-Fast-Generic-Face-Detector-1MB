"""
This code uses the pytorch model to detect faces from live video or camera.
"""
import argparse
import sys
import cv2

from vision.ssd.config.fd_config import define_img_size

parser = argparse.ArgumentParser(
    description='detect_video')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=640, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.7, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1000, type=int,
                    help='nms candidate size')
parser.add_argument('--path', default="imgs", type=str,
                    help='imgs dir')
parser.add_argument('--test_device', default="cuda:0", type=str,
                    help='cuda:0 or cpu')
parser.add_argument('--video_path', default="/home/cmf/tayg_duoren_part1.mp4", type=str,
                    help='path of video')
args = parser.parse_args()

input_img_size = args.input_size
define_img_size(
    input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer

label_path = "models/train-helmet-version-RFB-320/voc-model-labels.txt"

net_type = args.net_type

cap = cv2.VideoCapture(args.video_path)  # capture from video
# cap = cv2.VideoCapture(0)  # capture from camera

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = args.test_device

candidate_size = args.candidate_size
threshold = args.threshold

if net_type == 'slim':
    model_path = "models/pretrained/version-slim-320.pth"
    model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'RFB':
    model_path = "models/pretrained/version-RFB-320.pth"
    model_path = "models/pretrained/version-RFB-640.pth"
    model_path = 'models/train-dataset-oid-version-RFB/RFB-Epoch-90-Loss-3.197492961465877.pth'
    model_path = 'models/train-dataset-oid-version-RFB_v2/RFB-Epoch-1-Loss-3.1124141338097786.pth'
    model_path = 'models/train-dataset-oid-version-RFB_v3/RFB-Epoch-12-Loss-2.957992191732365.pth'
    model_path = 'models/train-dataset-oid-version-RFB-640/RFB-Epoch-75-Loss-2.86224332604095.pth'
    model_path = 'models/train-dataset-oid-version-RFB-640-pad/RFB-Epoch-63-Loss-2.7645139572394157.pth'
    model_path = 'models/train-dataset-oid-version-RFB-320-pad-without-resume/RFB-Epoch-199-Loss-2.9649808023967883.pth'
    model_path = 'models/train-helmet-version-RFB-320/RFB-Epoch-50-Loss-2.43834184328715.pth'
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

timer = Timer()
sum = 0

from vision.ssd.config.fd_config import image_size
from vision.transforms.transforms import Pad

# pad = Pad(image_size)


while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        print("end")
        break

    # orig_image, _, _ = pad(orig_image)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    timer.start()
    boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
    interval = timer.end()
    print('Time: {:.6f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    cv2.putText(orig_image, f'{1/interval:.2f} fps',
                (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 225), 2)
    for box, label, prob in zip(boxes, labels, probs):
        # box = boxes[i, :]
        # label = f" {probs[i]:.2f}"
        text = '{}_{:.2f}'.format(label, prob)
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)

        cv2.putText(orig_image, text,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    (0, 0, 255),
                    2)  # line type
    orig_image = cv2.resize(orig_image, None, None, fx=0.8, fy=0.8)
    sum += boxes.size(0)
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("all face num:{}".format(sum))

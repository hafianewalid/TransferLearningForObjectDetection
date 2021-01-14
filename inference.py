import argparse
import glob
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib
import os
from torchvision.ops import nms
import Models
import time
import Train
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2

parser = argparse.ArgumentParser()

parser.add_argument(
    '--image_file',
    type=str,
    help='Path to the image to test',
    required=False
)

parser.add_argument(
    '--images_set',
    type=str,
    help='Path to the images folder',
    required=False,
    default=""
)

parser.add_argument(
    '--model_file',
    type=str,
    help='The path to the trained model which will be used in inference',
    required=True
)

parser.add_argument(
    '--model_type',
    type=bool,
    help='True for SingleBboxHead, False for MultipleBboxHead',
    required=False,
    default=False
)

parser.add_argument(
    '--stream',
    type=bool,
    help='True for real time object detection, False for one image detection',
    required=False,
    default=False
)

parser.add_argument(
    '--cam',
    type=int,
    help='The device id used for real time image capturing (cam)',
    required=False,
    default=0
)

parser.add_argument(
    '--video',
    type=str,
    help='The input video',
    required=False,
    default=""
)

parser.add_argument(
    '--conf_threshold',
    type=float,
    help='The confidence threshold ',
    required=False,
    default=0.5
)

parser.add_argument(
    '--nms_threshold',
    type=float,
    help='The IOU threshold used in non maximum suppression phase',
    required=False,
    default=0.5
)

parser.add_argument(
    '--heat_map',
    type=bool,
    help='True to show the heat map associated to decision focus',
    required=False,
    default=False
)

args = parser.parse_args()

use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

classes = ['person', 'bird', 'cat', 'cow',
           'dog', 'horse', 'sheep', 'aeroplane',
           'bicycle', 'boat', 'bus', 'car',
           'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

info = pandas.read_csv(Train.features_path + '/info.csv')
model_tag = info.at[0, 'PertrainedModel']

global act


def Get_features(self, input, output):
    global act
    act = output.data


def draw_text(ax, x, y, text, color, backcolor):
    ax.text(x, y, text,
            horizontalalignment='left',
            verticalalignment='top', fontsize=12, color=color,
            bbox={'facecolor': 'black', 'alpha': 1, 'edgecolor': backcolor, 'linewidth': 4})


def object(predicted_bbox: torch.tensor, logits: torch.tensor, conf: float, ref: int, num_anchor: int):
    cx = (ref[0] + predicted_bbox[0][0].item()) / num_anchor
    cy = (ref[1] + predicted_bbox[0][1].item()) / num_anchor
    width = predicted_bbox[0][2].item()
    height = predicted_bbox[0][3].item()
    predicted_probas = F.softmax(logits, dim=1).squeeze()
    predicted_class = predicted_probas.argmax().item()
    bbox = torch.tensor([cx, cy, cx + width, cy + height])

    print("Bbox : (cx, cy, width, height) = ({},{},{},{})".format(cx, cy, width, height))
    print("Probabilities {}".format(predicted_probas))

    return (cx, cy, width, height, predicted_class, bbox, torch.tensor([conf]))


def preprocessing():
    # Preprocessing
    imagenet_preprocessing = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

    image_transform = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          imagenet_preprocessing])
    return image_transform


def load_model():
    feature_extractor = Models.pretrained(model_tag)
    if (type(feature_extractor) == tuple):
        feature_extractor = feature_extractor.model[0]

    # capture the first map activities
    if (args.heat_map):
        feature_extractor.conv1.register_forward_hook(Get_features)

    featuers_extracor = nn.Sequential(*list(feature_extractor.children())[:-2])
    classifier = torch.load(args.model_file)
    model = nn.Sequential(featuers_extracor, classifier)

    model = model.to(device=device)

    model.eval()

    return model


def image2tensor(image, image_transform: transforms):
    return image_transform(image).to(device=device).unsqueeze(0)


def inference(input_tensor: torch.tensor):
    with torch.no_grad():
        fps = time.time()
        all_predicted_bbox = []
        if args.model_type:
            predicted_bbox, logits = model(input_tensor)
            b = object(predicted_bbox, logits, 0, (0.0, 0.0), 1)
            all_predicted_bbox.append(b)
        else:
            predicted_bbox, logits, outputobj = model(input_tensor)
            objectnees = outputobj.permute(1, 2, 0)
            predicted_bbox = predicted_bbox.permute(2, 3, 0, 1)
            logits = logits.permute(2, 3, 0, 1)
            for i in range(7):
                for j in range(7):
                    p = F.softmax(logits[i][j], dim=1).squeeze().max().item()
                    conf = objectnees[i][j].item() + p
                    if (conf > args.conf_threshold):
                        box = object(predicted_bbox[i][j], logits[i][j], conf, (j, i), 7)
                        all_predicted_bbox.append(box)

            if (len(all_predicted_bbox) > 0):
                Bbox, Bconf = torch.stack([b[-2] for b in all_predicted_bbox]), torch.stack(
                    [b[-1] for b in all_predicted_bbox])
                keep_ind = nms(Bbox, Bconf, args.nms_threshold)
                all_predicted_bbox = [all_predicted_bbox[i] for i in keep_ind]

        fps = 1. / (time.time() - fps)

        return all_predicted_bbox, fps


def draw_prediction(image, all_predicted_bbox: list):
    plt.figure(0)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.imshow(image, extent=[0, 1, 1, 0])
    cm = plt.get_cmap('gist_rainbow')

    for cx, cy, width, height, predicted_class, tbox, conf in all_predicted_bbox:
        confs = str(round(conf.item() * 100, 2)) if (conf > 0) else ""
        class_color = cm(1. * predicted_class / len(classes))
        upl_x, upl_y = cx - width / 2.0, cy - height / 2.0
        p = patches.Rectangle((upl_x, upl_y), width, height, fill=False, clip_on=False, edgecolor=class_color,
                              linewidth=4)
        ax.add_patch(p)
        draw_text(ax, upl_x + 0.01, upl_y - 0.07, classes[predicted_class] + " " + confs, class_color, class_color)

    return plt


image_transform = preprocessing()
model = load_model()

if args.stream:

    if args.video == "":
        cap = cv2.VideoCapture(args.cam)
    else:
        cap = cv2.VideoCapture(args.video)

    while (cap.isOpened()):

        ret, image = cap.read()
        image = Image.fromarray(image[:, :, ::-1]).convert('RGB')
        input_tensor = image2tensor(image, image_transform)
        all_predicted_bbox, fps = inference(input_tensor)

        plt = draw_prediction(image, all_predicted_bbox)
        plt.savefig("od.png")
        plt.close()
        cv2.imshow('Object detection in the real time', cv2.imread('od.png', cv2.IMREAD_COLOR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    images_set = [args.image_file] if args.images_set == "" \
        else glob.glob(args.images_set + "/*.jpg")

    avg_fps = 0.
    for image_file in images_set:

        if not os.path.exists('Outputs'):
            os.makedirs('Outputs')
        output = "Outputs/" + ((image_file.split('/')[-1]).split('.')[0])

        image = Image.open(image_file).convert('RGB')
        input_tensor = image2tensor(image, image_transform)
        all_predicted_bbox, fps_ = inference(input_tensor)

        avg_fps += fps_
        plt = draw_prediction(image, all_predicted_bbox)
        plt.savefig(output + "_objects_prediction.png")
        plt.figure(0)
        plt.close()

        if (args.heat_map):
            plt.figure(1)
            act = act.squeeze()
            plt.imshow(torch.mean(act, 0) * torch.mean(act, 0))
            plt.savefig(output + '_Heat_Map.png')
            plt.figure(1)
            plt.close()

    avg_fps = avg_fps / len(images_set)
    file = open("results_fps.res", "a+")
    print(str(model_tag) + " " + str(int(avg_fps)), file=file)
    file.close()


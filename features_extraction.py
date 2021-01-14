import argparse
import os
import Loader
import Models
import torch
from collections import defaultdict
from tqdm import tqdm
import csv

parser = argparse.ArgumentParser()
parser.add_argument(
        '--model_type',
        type=bool,
        help='True for SingleBboxHead, False for MultipleBboxHead',
        required=False,
        default=False
)

parser.add_argument(
        '--pretrained',
        type=str,
        choices=['resnet18','resnet34','resnet50','resnet101','resnet152',
                 'alexnet','squeezenet','vgg16','densenet','inception',
                 'googlenet','shufflenet','mobilenet','mnasnet'],
        help='The pre-trained model used as features extraction',
        required=False,
        default='resnet18'
)

args = parser.parse_args()


def features_extraction_pm(loader: torch.utils.data.DataLoader,
                           model: torch.nn.Module,
                           device: torch.device,
                           filename: str):

    all_features = []
    all_targets = defaultdict(list)

    with torch.no_grad():
        model.eval()
        for (inputs, targets) in tqdm(loader):
            inputs = inputs.to(device=device)

            all_features.append(model(inputs))

            for k, v in targets.items():
                all_targets[k].append(v)


        for k, v in all_targets.items():
            all_targets[k] = torch.squeeze(torch.cat(v, 0))
        all_features = torch.squeeze(torch.cat(all_features , 0))

    print(all_features.shape)
    print("{} features maps ({} x {}, {})".format(all_features.shape[0], all_features.shape[2], all_features.shape[3], all_features.shape[1]))

    torch.save(dict([("features", all_features)] + list(all_targets.items())), filename)

    with open(filename.split('/')[0]+'/info.csv', mode='w') as info_file:
        info_writer = csv.writer(info_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        info_writer.writerow(['Maps_size', 'Channels', 'PertrainedModel'])
        info_writer.writerow([all_features.shape[2],all_features.shape[1],args.pretrained])


if args.model_type:
 print('largest_bbox')
 train_loader, valid_loader, test_loader= Loader.load_Pascal(targetmode='largest_bbox', imagemode='shrink')
 #train_loader, valid_loader= Load_data.load_Pascal(targetmode='preprocessed',imagemode='shrink')
else:
 print('all_bbox')
 train_loader, valid_loader, test_loader= Loader.load_Pascal(targetmode='all_bbox', imagemode='shrink', num_cell=7)

model = Models.Feature_Extracting(pretrained_name=args.pretrained)
model = model.to(device=Loader.device)


if not os.path.exists("features_data"):
    os.makedirs("features_data")

features_extraction_pm(train_loader, model, Loader.device, "features_data/train.txt")
features_extraction_pm(valid_loader, model, Loader.device, "features_data/valid.txt")
features_extraction_pm(test_loader, model, Loader.device, "features_data/test.txt")

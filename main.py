import argparse
import pandas
import torch
import Models
import Train


parser = argparse.ArgumentParser()

parser.add_argument(
        '--num_ep',
        type=int,
        help='Number of training epochs',
        required=False,
        default=100
        )
parser.add_argument(
        '--model_type',
        type=bool,
        help='True for SingleBboxHead, False for MultipleBboxHead',
        required=False,
        default=False
)
parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        required=False,
        default=0.001
)


args = parser.parse_args()

print(args.model_type)
tag="metrics/SingleBboxHead" if args.model_type else "metrics/MultipleBboxHead"

info=pandas.read_csv(Train.features_path+'/info.csv')
print(info)
map_size = info.at[0,'Maps_size']
channels = info.at[0,'Channels']
tag += "_"+info.at[0,'PertrainedModel']


model=Models.SingleBboxHead(channels * map_size * map_size, 20) if args.model_type else Models.MultipleBboxHead(channels, 20, 1)

f_loss1 = torch.nn.CrossEntropyLoss()#NLLLoss
f_loss2 = torch.nn.SmoothL1Loss()
f_loss3 = torch.nn.BCELoss()
f_loss4 = torch.nn.NLLLoss()


optimizer = torch.optim.Adam(model.parameters())

Train.train_nep_save(args.num_ep, tag, [f_loss1, f_loss2, f_loss3,f_loss4], optimizer, model)




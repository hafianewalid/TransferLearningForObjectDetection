import Data
import torch
import torchvision.transforms as transforms


use_gpu = torch.cuda.is_available()
if use_gpu:
        device = torch.device('cuda')
else:
        device = torch.device('cpu')

workers = 1
batch_size = 128

def load_Pascal(targetmode='largest_bbox',imagemode='shrink',num_cell=1):
        dataset_dir = "/opt/Datasets/Pascal-VOC2012/"
        download = False

        imagenet_preprocessing = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])

        image_transform = transforms.Compose([transforms.ToTensor(), imagenet_preprocessing])

        image_transform_params = {'image_mode':imagemode,'output_image_size': {'width': 224, 'height': 224}}
        target_transform_params = {'target_mode':targetmode,'image_transform_params': image_transform_params,'num_cells':num_cell}


        train_dataset, valid_dataset, test_dataset = Data.pascal_dataset(
                dataset_dir             = dataset_dir,
                image_transform_params  = image_transform_params,
                transform               = image_transform,
                target_transform_params = target_transform_params,
                download                = download
        )



        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=workers
                                                   )

        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=workers
                                                   )

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=workers
                                                  )

        return train_loader, valid_loader, test_loader


def load_features(path:str):

    datatrain = torch.load(path+'/train.txt')
    datavalid = torch.load(path+'/valid.txt')
    datatest = torch.load(path+'/test.txt')

    if(len(datatrain)==3):

        train_dataset = torch.utils.data.TensorDataset(datatrain['features'],
                                                       datatrain['bboxes'],
                                                       datatrain['labels'])

        valid_dataset =torch.utils.data.TensorDataset(datavalid['features'],
                                                      datavalid['bboxes'],
                                                      datavalid['labels'])

        test_dataset = torch.utils.data.TensorDataset(datatest['features'],
                                                       datatest['bboxes'],
                                                       datatest['labels'])

    else:

        train_dataset = torch.utils.data.TensorDataset(datatrain['features'],
                                                       datatrain['bboxes'],
                                                       datatrain['has_obj'],
                                                       datatrain['labels'])

        valid_dataset = torch.utils.data.TensorDataset(datavalid['features'],
                                                       datavalid['bboxes'],
                                                       datavalid['has_obj'],
                                                       datavalid['labels'] )

        test_dataset = torch.utils.data.TensorDataset(datatest['features'],
                                                       datatest['bboxes'],
                                                       datatest['has_obj'],
                                                       datatest['labels'])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)


    return train_loader,valid_loader,test_loader

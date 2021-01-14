import os
import math
import torch
import torchvision.transforms as transforms
import torchvision.datasets.voc as VOC

classes = ['person', 'bird', 'cat', 'cow',
           'dog', 'horse', 'sheep', 'aeroplane',
           'bicycle', 'boat', 'bus', 'car',
           'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']


def preprocess_target(target: dict):
    output_target = target.copy()

    output_target['annotation']['segmented'] = bool(output_target['annotation']['segmented'])
    for k, v in output_target['annotation']['size'].items():
        output_target['annotation']['size'][k] = int(v)

    if type(output_target['annotation']['object']) is not list:
        output_target['annotation']['object'] = [output_target['annotation']['object']]

    objects = output_target['annotation']['object']

    for o in objects:
        for k, v in o['bndbox'].items():
            o['bndbox'][k] = int(v)
        for k in ['occluded', 'difficult', 'truncated']:
            o[k] = bool(int(o[k]))

    return output_target


def extract_class_and_bndbox(target: dict, image_transform_params: dict):
    return [{'bndbox': transform_bbox(o['bndbox'],
                                      {'width': target['annotation']['size']['width'],
                                       'height': target['annotation']['size']['height']},
                                      image_transform_params),
             'class': classes.index(o['name'])}
            for o in target['annotation']['object']]


def transform_bbox(bbox: dict, input_image_size: dict, image_transform_params: dict):
    out_bbox = {"cx": 0.0, "cy": 0.0, "width": 0.0, "height": 0.0}

    image_mode = image_transform_params['image_mode']
    if image_mode == 'none':
        out_bbox["cx"] = 0.5 * (bbox['xmin'] + bbox['xmax']) / input_image_size['width']
        out_bbox["cy"] = 0.5 * (bbox['ymin'] + bbox['ymax']) / input_image_size['height']
        out_bbox["width"] = float(bbox["xmax"] - bbox["xmin"]) / input_image_size["width"]
        out_bbox["height"] = float(bbox["ymax"] - bbox["ymin"]) / input_image_size["height"]

    elif (image_mode == 'shrink'):
        output_image_size = image_transform_params['output_image_size']
        scale_width = float(output_image_size['width']) / input_image_size['width']
        scale_height = float(output_image_size['height']) / input_image_size['height']
        out_bbox["cx"] = scale_width * 0.5 * (bbox['xmin'] + bbox['xmax']) / output_image_size['width']
        out_bbox["cy"] = scale_height * 0.5 * (bbox['ymin'] + bbox['ymax']) / output_image_size['height']
        out_bbox["width"] = scale_width * float(bbox["xmax"] - bbox["xmin"]) / output_image_size["width"]
        out_bbox["height"] = scale_height * float(bbox["ymax"] - bbox["ymin"]) / output_image_size["height"]

    elif (image_mode == 'crop'):
        output_image_size = image_transform_params['output_image_size']
        offset_width = int(round((input_image_size['width'] - output_image_size['width']) / 2.))
        offset_height = int(round((input_image_size['height'] - output_image_size['height']) / 2.))

        cropped_bbox = {"xmin": 0.0, "xmax": 0.0, "ymin": 0.0, "ymax": 0.0}
        for sfx in ['min', 'max']:
            cropped_bbox['x%s' % sfx] = min(max(bbox['x%s' % sfx] - offset_width, 0), output_image_size['width'])
            cropped_bbox['y%s' % sfx] = min(max(bbox['y%s' % sfx] - offset_height, 0), output_image_size['height'])
        out_bbox["cx"] = 0.5 * (cropped_bbox['xmin'] + cropped_bbox['xmax']) / output_image_size['width']
        out_bbox["cy"] = 0.5 * (cropped_bbox['ymin'] + cropped_bbox['ymax']) / output_image_size['height']
        out_bbox["width"] = float(cropped_bbox["xmax"] - cropped_bbox["xmin"]) / output_image_size["width"]
        out_bbox["height"] = float(cropped_bbox["ymax"] - cropped_bbox["ymin"]) / output_image_size["height"]

    else:
        raise ValueError('invalid image_mode for transform_bbox, got "{}"'.format(image_mode))
    return out_bbox


def filter_largest(objects: list):
    largest, max_size = {}, 0
    for obj in objects:
        size = obj['bndbox']['width'] * obj['bndbox']['height']
        if (size > max_size):
            largest, max_size = obj, size

    return largest


def target_to_tensor(obj: dict):
    tensor1 = torch.Tensor([obj['bndbox']['cx'], obj['bndbox']['cy'], obj['bndbox']['width'], obj['bndbox']['height']])
    tensor2 = torch.LongTensor([obj['class']])
    return {'bboxes': tensor1, 'labels': tensor2}


def cell_idx_of_center(coordinates, num_cells: int):
    return math.floor(coordinates[0] * num_cells), math.floor(coordinates[1] * num_cells)


def targets_to_grid_cell_tensor(objects: list, num_cells: int):
    bboxes = torch.zeros((4, num_cells, num_cells), dtype=torch.float)
    has_obj = torch.zeros((num_cells, num_cells), dtype=torch.int)
    labels = torch.zeros((num_cells, num_cells), dtype=torch.int)

    for ko, o in enumerate(objects):
        bndbox = o['bndbox']
        cx, cy, width, height = bndbox['cx'], bndbox['cy'], bndbox['width'], bndbox['height']
        cj, ci = cell_idx_of_center((cx, cy), num_cells)

        Xincell, Yincell = cx * num_cells - cj, cy * num_cells - ci
        bboxes[:, ci, cj] = torch.Tensor([Xincell, Yincell, width, height])
        has_obj[ci, cj] = 1
        labels[ci, cj] = o['class']

    return {'bboxes': bboxes, 'has_obj': has_obj, 'labels': labels}


def check_key(d, key, valid_values):
    if not key in d:
        raise KeyError('Missing key {} in dictionnary {}'.format(key, d))
    if not d[key] in valid_values:
        raise ValueError("Key {}: got \"{}\" , expected one of {}".format(key, d[key], valid_values))


def validate_image_transform_params(image_transform_params: dict):
    check_key(image_transform_params, 'image_mode', ['none', 'shrink', 'crop'])

    if (image_transform_params['image_mode'] == 'none'):
        return
    else:
        assert ('output_image_size' in image_transform_params)
        assert (type(image_transform_params['output_image_size']) is dict)
        assert ('width' in image_transform_params['output_image_size'])
        assert ('height' in image_transform_params['output_image_size'])


def make_image_transform(image_transform_params: dict,
                         transform: object):
    validate_image_transform_params(image_transform_params)

    resize_image = image_transform_params['image_mode']
    if resize_image == 'none':
        preprocess_image = None
    elif resize_image == 'shrink':
        preprocess_image = transforms.Resize((image_transform_params['output_image_size']['width'],
                                              image_transform_params['output_image_size']['height']))
    elif resize_image == 'crop':
        preprocess_image = transforms.CenterCrop((image_transform_params['output_image_size']['width'],
                                                  image_transform_params['output_image_size']['height']))

    if preprocess_image is not None:
        if transform is not None:
            image_transform = transforms.Compose([preprocess_image, transform])
        else:
            image_transform = preprocess_image
    else:
        image_transform = transform

    return image_transform


def validate_target_transforms_params(target_transform_params: dict):
    check_key(target_transform_params, 'target_mode', ['orig', 'preprocessed', 'largest_bbox', 'all_bbox'])

    if (target_transform_params['target_mode'] in ['orig', 'preprocessed']):
        return
    else:
        assert ('image_transform_params' in target_transform_params)
        assert (type(target_transform_params['image_transform_params']) is dict)
        validate_image_transform_params(target_transform_params['image_transform_params'])
        if (target_transform_params['target_mode'] == 'all_bbox'):
            assert ('num_cells' in target_transform_params)


def make_target_transform(target_transform_params: dict):
    validate_target_transforms_params(target_transform_params)

    target_mode = target_transform_params['target_mode']

    if target_mode == 'orig':
        return None
    elif target_mode == 'preprocessed':
        t_transform = lambda target: preprocess_target(target)
    else:
        image_transform_params = target_transform_params['image_transform_params']
        get_bbox = lambda target: extract_class_and_bndbox(preprocess_target(target), image_transform_params)
        if target_mode == 'largest_bbox':
            t_transform = lambda target: target_to_tensor(filter_largest(get_bbox(target)))
        else:
            t_transform = lambda target: targets_to_grid_cell_tensor(get_bbox(target),
                                                                     target_transform_params['num_cells'])
    return t_transform


def pascal_dataset(dataset_dir: str,
                   image_transform_params: dict,
                   transform: object,
                   target_transform_params: dict,
                   download: bool):
    if not dataset_dir:
        dataset_dir = os.path.join(os.path.expanduser("~"), 'Datasets', 'PascalVOC')

    image_transform = make_image_transform(image_transform_params, transform)
    target_transform = make_target_transform(target_transform_params)

    dataset_train = VOC.VOCDetection(root=dataset_dir, image_set='train',
                                     transform=image_transform,
                                     target_transform=target_transform,
                                     download=download)
    dataset_val = VOC.VOCDetection(root=dataset_dir, image_set='val',
                                   transform=image_transform,
                                   target_transform=target_transform,
                                   download=download)
    dataset_test = VOC.VOCDetection(root=dataset_dir, image_set='val',
                                    transform=image_transform,
                                    target_transform=target_transform,
                                    download=download)

    return dataset_train, dataset_val, dataset_test


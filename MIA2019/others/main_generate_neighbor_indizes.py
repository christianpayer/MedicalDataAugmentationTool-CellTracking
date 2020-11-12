#!/usr/bin/python
import os

from dataset_for_generate_neighbor_indizes import Dataset
import utils.io.text
import utils.io.image
import numpy as np
from utils.np_image import center_of_mass, draw_circle, dilation_circle


def get_dataset_parameters(dataset_name):
    instance_image_radius_factors = {'DIC-C2DH-HeLa': 0.2,
                                     'Fluo-C2DL-MSC': 0.6,
                                     'Fluo-N2DH-GOWT1': 0.2,
                                     'Fluo-N2DH-SIM+': 0.2,
                                     'Fluo-N2DL-HeLa': 0.03,
                                     'PhC-C2DH-U373': 0.2,
                                     'PhC-C2DL-PSC': 0.03}
    return {'instance_image_radius_factor': instance_image_radius_factors[dataset_name]}


def image_sizes_for_dataset_name(dataset_name):
    test_image_size = {'DIC-C2DH-HeLa': [256, 256],  # * 2
                        'Fluo-C2DL-MSC':  [384, 256],  # uneven
                        'Fluo-N2DH-GOWT1': [512, 512],  # * 2
                        'Fluo-N2DH-SIM+': [512, 512],  # uneven
                        'Fluo-N2DL-HeLa': [1024, 640],  # exact
                        'PhC-C2DH-U373': [512, 384],  # uneven
                        'PhC-C2DL-PSC': [720, 576]}  # exact
    return test_image_size[dataset_name]


def current_instance_neighbors(target_label, label_image, image_size, instance_image_radius_factor, loss_mask_dilation_size):
    """
    Returns the mask, of neighboring other_labels for a given target_label.
    :param target_label: Image, where pixels of the target label are set to 1, all other pixels are 0.
    :param other_labels: List of images of all labels. Pixels of the label are set to 1, all other pixels are 0.
    :return: The image (np array), where pixels of neighboring labels are set to 2, all other pixels are 0.
    """
    frame_axis = 0
    dim = len(label_image.shape)

    mask = (label_image == target_label).astype(np.uint8)
    if loss_mask_dilation_size > 0:
        if dim == 3:
            for i in range(label_image.shape[frame_axis]):
                current_slice = [slice(None), slice(None)]
                current_slice.insert(frame_axis, slice(i, i + 1))
                mask[tuple(current_slice)] = dilation_circle(np.squeeze(mask[tuple(current_slice)]), (loss_mask_dilation_size, loss_mask_dilation_size))
        else:
            mask = dilation_circle(mask, (loss_mask_dilation_size, loss_mask_dilation_size))
    circle_radius = image_size[0] * instance_image_radius_factor
    # handle images with dim 3 (videos) differently
    if dim == 3:
        com = center_of_mass(mask)
        com = com[1:]
        is_in_frame = np.any(np.any(mask, axis=1), axis=1)
        min_index = np.maximum(np.min(np.where(is_in_frame)) - 1, 0)
        max_index = np.minimum(np.max(np.where(is_in_frame)) + 2, label_image.shape[frame_axis])
        for i in range(min_index, max_index):
            current_slice = [slice(None), slice(None)]
            current_slice.insert(frame_axis, slice(i, i + 1))
            if i == min_index:
                next_slice = [slice(None), slice(None)]
                next_slice.insert(frame_axis, slice(i + 1, i + 2))
                mask[tuple(current_slice)] = np.logical_or(mask[tuple(next_slice)], mask[tuple(current_slice)])
            if i == max_index - 1:
                previous_slice = [slice(None), slice(None)]
                previous_slice.insert(frame_axis, slice(i - 1, i))
                mask[tuple(current_slice)] = np.logical_or(mask[tuple(previous_slice)], mask[tuple(current_slice)])
            current_mask = np.squeeze(mask[tuple(current_slice)], axis=frame_axis)
            draw_circle(current_mask, com, circle_radius)
    else:
        com = center_of_mass(label_image)
        draw_circle(mask, com, circle_radius)
    current_neighbors = [int(x) for x in np.unique(label_image * mask) if x != 0 and x != target_label]
    return current_neighbors


def instance_neighbors(image, image_size, instance_image_radius_factor, loss_mask_dilation_size=5):
    """
    Returns the stacked instance images for the current instance segmentation. The resulting np array contains
    images for instances that are stacked at the channel axis. Each entry of the channel axis corresponds to
    a certain instance, where pixels with value 1 indicate the instance, 2 other neighboring instances, and 0 background.
    :param image: The groundtruth instance segmentation.
    :param instances_datatype: The np datatype for the bitwise instance image.
    :return: The stacked instance image.
    """
    label_indizes = np.unique(image)
    neighbors = {}
    for label in label_indizes:
        if label == 0:
            continue
        current_neighbors = current_instance_neighbors(target_label=label,
                                                       label_image=image,
                                                       image_size=image_size,
                                                       instance_image_radius_factor=instance_image_radius_factor,
                                                       loss_mask_dilation_size=loss_mask_dilation_size)
        neighbors[int(label)] = current_neighbors
    return neighbors


def generate_dataset_neighbors(dataset_name):
    print('processing dataset', dataset_name)
    test_image_size = image_sizes_for_dataset_name(dataset_name)

    data_format = 'channels_first'
    dataset_name = dataset_name
    base_folder = os.path.join('../../celltrackingchallenge/trainingdataset/', dataset_name)

    # train dataset
    dataset_parameters = get_dataset_parameters(dataset_name)
    dataset_parameters.update({'image_size': test_image_size,
                               'base_folder': base_folder,
                               'data_format': data_format})
    dataset = Dataset(**dataset_parameters)

    video_ids = ['01', '02']
    frames = utils.io.text.load_dict_csv('../../celltrackingchallenge/trainingdataset/' + dataset_name + '/setup/frames.csv')

    dataset_single_frame = dataset.dataset_single_frame()
    output_folder = '../../celltrackingchallenge/trainingdataset/' + dataset_name + '/setup/'

    neighbors = {}
    for video_id in video_ids:
        seg = []
        for frame_id in frames[video_id]:
            id_dict = {'video_id': video_id, 'frame_id': frame_id}
            current_frame_data = dataset_single_frame.get(id_dict)
            seg.append(current_frame_data['generators']['merged'])
        seg = np.concatenate(seg, axis=0)
        current_neighbors = instance_neighbors(seg, test_image_size, dataset_parameters['instance_image_radius_factor'])
        neighbors[video_id] = current_neighbors
    utils.io.text.save_json(neighbors, os.path.join(output_folder, 'instance_neighbors.json'))


if __name__ == '__main__':
    datasets = ['DIC-C2DH-HeLa',
                'Fluo-N2DL-HeLa',
                'Fluo-N2DH-GOWT1',
                'Fluo-N2DH-SIM+',
                'PhC-C2DH-U373',
                'Fluo-C2DL-MSC']
    for dataset in datasets:
        generate_dataset_neighbors(dataset)

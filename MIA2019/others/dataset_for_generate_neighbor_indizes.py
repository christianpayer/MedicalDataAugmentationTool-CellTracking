import SimpleITK as sitk
import numpy as np
from datasets.graph_dataset import GraphDataset
from datasources.image_datasource import ImageDataSource
from generators.image_generator import ImageGenerator
from transformations.spatial import translation, scale, composite


class Dataset(object):
    """
    The dataset that processes files from the celltracking challenge.
    """
    def __init__(self,
                 image_size,
                 base_folder='',
                 data_format='channels_first',
                 loss_mask_dilation_size=5,
                 instance_image_radius_factor=0.2,
                 image_interpolator='linear'):
        self.image_size = image_size
        self.image_base_folder = base_folder
        self.loss_mask_dilation_size = loss_mask_dilation_size
        self.instance_image_radius_factor = instance_image_radius_factor
        self.data_format = data_format
        self.image_interpolator = image_interpolator
        self.dim = 2

    def get_channel_axis(self, image, data_format):
        """
        Returns the channel axis of the given image.
        :param image: The np array.
        :param data_format: The data format. Either 'channels_first' or 'channels_last'.
        :return: The channel axis.
        """
        if len(image.shape) == 3:
            return 0 if data_format == 'channels_first' else 2
        if len(image.shape) == 4:
            return 0 if data_format == 'channels_first' else 3

    def datasources_single_frame(self, iterator):
        """
        Returns the data sources that load data for a single frame.
        {
        'merged:' CachedImageDataSource that loads the segmentation/tracking label files.
        }
        :return: A dict of data sources.
        """
        datasources_dict = {}
        # image data source loads input image.
        datasources_dict['merged'] = ImageDataSource(self.image_base_folder,
                                                     file_ext='.mha',
                                                     set_identity_spacing=True,
                                                     sitk_pixel_type=sitk.sitkUInt16,
                                                     id_dict_preprocessing=lambda x: {'image_id': x['video_id'] + '_GT/MERGED/' + x['frame_id']},
                                                     name='merged',
                                                     parents=[iterator])
        return datasources_dict

    def data_generators_single_frame(self, dim, datasources, image_transformation):
        """
        Returns the data generators that process a single frame. See datasources_single_frame() for dict values.
        :param dim: Image dimension.
        :param image_post_processing: The np postprocessing function for the image data generator.
        :return: A dict of data generators.
        """
        image_size = self.image_size
        data_generators_dict = {}
        data_generators_dict['merged'] = ImageGenerator(dim, image_size,
                                                        interpolator='nearest',
                                                        data_format=self.data_format,
                                                        np_pixel_type=np.uint16,
                                                        name='merged',
                                                        parents=[datasources['merged'], image_transformation])
        return data_generators_dict

    def spatial_transformation(self, image):
        """
        The spatial image transformation without random augmentation.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    scale.Fit(self.dim, self.image_size),
                                    translation.OriginToOutputCenter(self.dim, self.image_size)],
                                   name='image_transformation',
                                   kwparents={'image': image})

    def dataset_single_frame(self):
        """
        Returns the training dataset for single frames. Random augmentation is performed.
        :return: The training dataset.
        """
        iterator = 'iterator'
        sources = self.datasources_single_frame(iterator)
        image_key = 'merged'
        image_transformation = self.spatial_transformation(sources[image_key])
        generators = self.data_generators_single_frame(2, sources, image_transformation)

        return GraphDataset(data_generators=list(generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[image_transformation],
                            iterator=iterator,
                            debug_image_folder=None)

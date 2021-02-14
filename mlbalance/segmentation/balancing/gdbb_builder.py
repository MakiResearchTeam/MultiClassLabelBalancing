# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import absolute_import
import pandas as pd
import cv2
from .. import ElasticAugment, Data
from .utils import load_json
import os


class GD2BBuilder:
    def __init__(self, mask_labelsetid, balance_config, masks_images, resize=None, use_augment=True):
        """
        Parameters
        ----------
        mask_labelsetid : str or dict
            Json containing pairs { masks_name: labelset_id }
        balance_config : str or dict
            Json containing pairs { labelset_id: n_copies }
        masks_images : str or dict
            Json containing pairs { path_to_mask : path_to_image }.

        resize : tuple
            Images will be resized accordingly while loading.
        """
        self._maskname_labelsetid_d = mask_labelsetid if isinstance(mask_labelsetid, dict) else load_json(mask_labelsetid)
        self._balance_config = balance_config if isinstance(balance_config, dict) else load_json(balance_config)
        self.use_augment = use_augment

        masks_images = masks_images if isinstance(masks_images, dict) else load_json(masks_images)
        self._resize = resize
        self._load_masks_images(masks_images, resize)

        self._group_images_masks_by_id()
        self._aug = None

    # noinspection PyAttributeOutsideInit
    def _load_masks_images(self, masks_images: dict, resize):
        print('Loading masks and images.')
        self._images_masks = {}
        for mask_name, image_name in masks_images.items():
            image = cv2.imread(image_name)
            assert image is not None
            mask = cv2.imread(mask_name)
            assert mask is not None

            if resize is not None:
                image = cv2.resize(image, resize, interpolation=cv2.INTER_CUBIC)
                mask = cv2.resize(mask, resize, interpolation=cv2.INTER_NEAREST)

            self._images_masks[mask_name] = (image, mask)
        print('Finished.')

    def _group_images_masks_by_id(self):
        print('Group masks and images by their ids.')
        self._labelset_ids = {}
        for maskname, labelset_id in self._maskname_labelsetid_d.items():
            self._labelset_ids[labelset_id] = [self._images_masks[maskname]] + self._labelset_ids.get(labelset_id, [])

        for labelset_id in self._labelset_ids:
            print(f'{labelset_id} cardinality is {len(self._labelset_ids[labelset_id])}')
        print('Finished.')

    # noinspection PyAttributeOutsideInit
    def set_elastic_aug_params(
            self, img_shape=None,
            alpha=500, std=8, noise_invert_scale=5,
            img_inter='linear', mask_inter='nearest', border_mode='reflect'
    ):
        self._aug_alpha = alpha
        self._aug_std = std
        self._aug_noise_invert_scale = noise_invert_scale
        self._aug_img_inter = img_inter
        self._aug_mask_inter = mask_inter
        self._aug_border_mode = border_mode
        if self._resize is not None:
            self._img_shape = self._resize
        else:
            self._img_shape = img_shape
        assert self._img_shape is not None

    def _create_augment(self):
        self._aug = ElasticAugment(
            alpha=self._aug_alpha,
            std=self._aug_std,
            num_maps=1,
            noise_invert_scale=self._aug_noise_invert_scale,
            img_inter=self._aug_img_inter,
            mask_inter=self._aug_mask_inter,
            border_mode=self._aug_border_mode
        )
        self._aug.setup_augmentor(self._img_shape)

    def create_batch(self, path_to_save):
        """
        path_to_save : str
            Path to the folder where the results of the data processing will be saved.
            Example: '.../balanced_batch'.
        """
        for labelset_id in self._labelset_ids:
            imgs, masks = self._balance_group(labelset_id)
            self._save_imgs(imgs, masks, labelset_id, path_to_save)
            print(f'{labelset_id} ready')

    def _save_imgs(self, imgs, masks, hcv_group, path_to_save):
        masks_path = os.path.join(path_to_save, 'masks')
        imgs_path = os.path.join(path_to_save, 'images')
        os.makedirs(masks_path, exist_ok=True)
        os.makedirs(imgs_path, exist_ok=True)
        for i, (img, mask) in enumerate(zip(imgs, masks)):
            cv2.imwrite(masks_path+f'/{hcv_group}_{i}.bmp', mask)
            cv2.imwrite(imgs_path + f'/{hcv_group}_{i}.bmp', img)

    def _balance_group(self, labelset_id):
        print(f'Balancing group {labelset_id}...')
        imgs, masks = [], []
        img_ind = 0
        aug_updates = 0
        labelset_ncopies = self._balance_config[labelset_id]
        while labelset_ncopies > 0:
            im, mask = self._labelset_ids[labelset_id][img_ind]
            if self.use_augment:
                im, mask = self._augment(im, mask)
            imgs.append(im)
            masks.append(mask)
            labelset_ncopies -= 1
            img_ind += 1
            if img_ind == len(self._labelset_ids[labelset_id]):
                img_ind = 0
                self._create_augment()
                aug_updates += 1

        self._aug = None
        print(f'Augmentor updated {aug_updates} times.')
        print(f'Finished.')
        return imgs, masks

    def _augment(self, im, mask):
        if self._aug is None:
            return im, mask
        data = Data(images=[im], masks=[mask])
        im, mask = self._aug(data).get_data()
        return im[0], mask[0]

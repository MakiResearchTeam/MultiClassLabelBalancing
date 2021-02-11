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
from .utils import hcv_to_num, to_hc_vec, has_classes
import numpy as np
import cv2
import pandas as pd


# Has-Class Scanner
class HCScanner:
    def __init__(self, masks, num_classes):
        """
        Parameters
        ----------
        masks: dictionary or list
            Dictionary case: contains pairs 'mask name : mask'.
            List case: contains paths to the masks.
        """
        if isinstance(masks, list):
            self.__load_masks(masks)
        else:
            self.filename_mask_d = masks
        self.num_classes = num_classes
        self.__scan()

    def __load_masks(self, paths):
        self.filename_mask_d = {}
        for path in paths:
            self.filename_mask_d[path] = cv2.imread(path)

    def __scan(self):
        # { filename : labelset_id }
        self.filename_labelsetid_d = {}
        # { labelset_id : number of vectors }
        self.labelsetid_nimages_d = {}
        # { labelset_id: labelset } - contains only unique labelsets
        self.labelsetid_labelset_d = {}
        for filename, mask in self.filename_mask_d.items():
            classes = np.unique(mask)
            labelset = to_hc_vec(self.num_classes, classes)
            labelset_id = hcv_to_num(labelset)

            self.filename_labelsetid_d[filename] = labelset_id
            self.labelsetid_nimages_d[labelset_id] = 1 + self.labelsetid_nimages_d.get(labelset_id, 0)
            if self.labelsetid_nimages_d[labelset_id] == 1:
                self.labelsetid_labelset_d[labelset_id] = labelset

    def get_labelsets(self):
        """
        Returns
        -------
        ndarray of shape [n_labelsets, n_classes + 1]
            A matrix which first n_classes columns is a set of labelsets. The last column contains
            number of images that correspond to the labelsets in the corresponding rows.
        """
        labelsets = []
        for labelset_id, n_copies in self.labelsetid_nimages_d.items():
            labelset = self.labelsetid_labelset_d[labelset_id]
            labelset = np.concatenate([labelset, [n_copies]], axis=0)
            labelsets.append(labelset)
        return np.asarray(labelsets)

    def save_labelsets(self, path):
        """
        Saves the matrix generated by the `get_labelsets` method.

        Parameters
        ----------
        path : str
            Example: 'labelsets.npy'
        """
        labelsets = self.get_labelsets()
        np.save(path, labelsets)

    def select_masks_by_classes(self, classes):
        """
        Parameters
        ----------
        classes : list
            A list of classes a mask has to contain. Example: [0, 4, 2].

        Returns
        -------
        dict { filename: mask }
            Masks that contain the classes.
        """
        chosen_labelset_ids = []
        for labelset_id, labelset in self.labelsetid_labelset_d.items():
            if has_classes(labelset, classes):
                chosen_labelset_ids.append(labelset_id)

        masks = {}
        for labelset_id in chosen_labelset_ids:
            masks.update(self.select_masks_by_labelset_id(labelset_id))

        return masks

    def select_masks_by_labelset_id(self, labelset_id):
        """
        Looks up for masks with the given labelset_id.

        Parameters
        ----------
        labelset_id : int
            A labelset_id.

        Returns
        -------
        dict { filename: mask }
            Masks with the corresponding labelset_id.
        """
        masks = {}
        for filename, labelset_id_ in self.filename_labelsetid_d.items():
            if labelset_id_ == labelset_id:
                masks[filename] = self.filename_mask_d[filename]

        return masks

    def get_class_frequencies(self):
        labelsets = self.get_labelsets()
        labelsets, n_masks = labelsets[:, :-1], labelsets[:, -1:]
        freq = np.sum(labelsets * n_masks, axis=0) / np.sum(n_masks)
        return freq

    def save_info(self, uniq_hvc_path, masks_hcvg_path):
        pd.DataFrame.from_dict(self.labelsetid_labelset_d, orient='index').to_csv(uniq_hvc_path)
        pd.DataFrame.from_dict(self.filename_labelsetid_d, orient='index', columns=['hcvg']).to_csv(masks_hcvg_path)
        print('Saved!')
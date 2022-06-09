import SimpleITK as sitk
import cv2, os, json
from tqdm import tqdm
import numpy as np

# libraries
import os
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib.pyplot as plt

# custom
from data_utilities import *


def pack_pointsList(json_path):
    """
    :param coords: coords in json, dict list
    :return: list, [[x,y,z], ...]
    """
    coord_list = []
    with open(json_path, 'r') as f:
        json_obj = json.load(f)
        for i in range(1, len(json_obj)):
            pt = []
            print(json_obj[i]["label"])
            pt.append(json_obj[i]["X"])
            pt.append(json_obj[i]["Y"])
            pt.append(json_obj[i]["Z"])
            coord_list.append(pt)

    return coord_list


class HeatmapGenerator():
    def __init__(self, sigma: float = 5.0):
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, h, w, l, points):
        points = pack_pointsList(points)
        hms = np.zeros([h, w], dtype=np.float32)
        sigma = self.sigma
        for pt in points:
            x, y, z = int(pt[0]), int(pt[1]), int(pt[2])
            if x < 0 or y < 0 or z < 0 or \
                    x >= w or y >= h or z >= l:
                continue

            ul = int(np.round(x - 3 * sigma - 1)
                     ), int(np.round(y - 3 * sigma - 1)), int(np.round(z - 3 * sigma - 1))
            br = int(np.round(x + 3 * sigma + 2)
                     ), int(np.round(y + 3 * sigma + 2)), int(np.round(z + 3 * sigma + 2))

            c, d = max(0, -ul[0]), min(br[0], w) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], h) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], w)
            aa, bb = max(0, ul[1]), min(br[1], h)
            hms[aa:bb, cc:dd] = \
                np.maximum(hms[aa:bb, cc:dd], self.g[a:b, c:d])

            return hms


def save_images(img):
    for j in tqdm(range(3)):
        os.mkdir('{0}'.format(j))
        for i in range(img.shape[j]):
            if j == 0:
                cv2.imwrite('.\{0}\{1}.jpg'.format(j, i), img.__mul__(30).transpose((0, 1, 2))[i])
            if j == 1:
                cv2.imwrite('.\{0}\{1}.jpg'.format(j, i), img.__mul__(30).transpose((1, 0, 2))[i])
            if j == 2:
                cv2.imwrite('.\{0}\{1}.jpg'.format(j, i), img.__mul__(30).transpose((2, 0, 1))[i])


# def display():
#     fig = plt.figure()
#     ax = Axes3D(fig)


if __name__ == '__main__':
    img_nib = nib.load("E:\Verse\dataset-01training\derivatives\sub-gl003\sub-gl003_dir-ax_seg-vert_msk.nii.gz")
    ctd_list = load_centroids("E:\Verse\dataset-01training\derivatives\sub-gl003\sub-gl003_dir-ax_seg-subreg_ctd.json")
    axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine))
    print('img orientation code: {}'.format(axs_code))
    print('Centroid List: {}'.format(ctd_list))
    img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=3)
    img_iso = reorient_to(img_iso, axcodes_to=('I', 'P', 'L'))
    ctd_iso = rescale_centroids(ctd_list, img_nib, (1, 1, 1))
    # get vocel data
    im_np = img_iso.get_fdata()
    zooms = img_iso.header.get_zooms()
    # save_images(im_np)

    # get the mid-slice of the scan and mask in both sagittal and coronal planes

    im_np_sag = im_np[:, :, int(im_np.shape[2] / 2)]
    im_np_cor = im_np[:, int(im_np.shape[1] / 2), :]
    # im_np_sag = cv2.cvtColor(im_np_sag, cv2.COLOR_RGB2GRAY)
    # im_np_sag = cv2.imread('im_np_sag.jpg', cv2.CV_8U)
    cv2.imwrite('im_np_sag.jpg',im_np_sag)
    # cv2.waitKey(0)
    # im_np_cor = cv2.cvtColor(im_np_cor, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('im_np_cor', im_np_cor)
    # cv2.waitKey(0)

    # plot
    fig, axs = create_figure(96, im_np_sag, im_np_cor)

    axs[0].imshow(im_np_sag, cmap=plt.cm.gray, norm=wdw_sbone)
    plot_sag_centroids(axs[0], ctd_iso, zooms)

    axs[1].imshow(im_np_cor, cmap=plt.cm.gray, norm=wdw_sbone)
    plot_cor_centroids(axs[1], ctd_iso, zooms)
    a=1
# HG = HeatmapGenerator(sigma=5)
# heatmap = HG()

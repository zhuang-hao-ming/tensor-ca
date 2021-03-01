# -*- coding: utf-8 -*-

import sys


from scipy import signal
import cupy as cp
import rasterio




def count_neighbour_vectorize(img, moor_radius=3, weight_of_neighbour=[1, 1, 1, 1], land_use_count=4):
    '''
    Using vectorization method to count the number of cell of different land use types around each cell
    

    Parameters:
    -------------------
    img: cupy array
        land use array
    moor_radius: int
        size of moor radius
    weight_of_neighbour: array
        neighborhood weight of each land use type
    land_use_count: int
        number of land use types

    
    Returns:
    ------------------

    return a ndarray
    result[:,:,land_type] neighborhood probability of land use type k

    '''

    dead_span = moor_radius // 2
    new_m = img.shape[0] + (dead_span * 2)
    new_n = img.shape[1] + (dead_span * 2)



    neighbour_img_list = []
    for index in range(land_use_count):
        land_type = index + 1

        landuse_img = cp.zeros((new_m, new_n), dtype=cp.int8)
        landuse_img[dead_span:-dead_span,dead_span:-dead_span] = (img == land_type).astype(cp.int8)

        neighbour_img = cp.zeros(img.shape, dtype=cp.float16)
        for row in range(0, moor_radius):
            for col in range(0, moor_radius):
                if row == dead_span and col == dead_span:
                    continue
                row_end = new_m - (moor_radius - 1 - row)
                col_end = new_n - (moor_radius - 1 - col)
                neighbour_img += landuse_img[row:row_end, col:col_end]

        neighbour_img = neighbour_img / (moor_radius*moor_radius-1) * weight_of_neighbour[index]
        neighbour_img_list.append(neighbour_img)

    neighbour_img = cp.stack(neighbour_img_list, axis=2)
    return neighbour_img
















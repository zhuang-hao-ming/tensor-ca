# -*- coding: utf-8 -*-

import sys


import datetime
from time import time

import numpy as np
import rasterio
import rasterio.shutil

import cupy as cp
from count_neighbour_gpu import count_neighbour_vectorize

# cp.random.seed(100)

# # Disable memory pool for device memory (GPU)
# cp.cuda.set_allocator(None)

# # Disable memory pool for pinned memory (CPU).
# cp.cuda.set_pinned_memory_allocator(None)

# mempool = cp.get_default_memory_pool()
# pinned_mempool = cp.get_default_pinned_memory_pool()

def read_img(img_path):
    '''
    read tiff image

    Parameters:
    -------------------
    img_path: str
        path of image
    padding: dict
        valid area of image, 
    
    Returns:
    -----------------
    band: ndarray
        image ndarray
    con：ndarray
        valida data ndarray
    '''
    with rasterio.open(img_path) as dst:
        count = dst.count
        width = dst.width
        height = dst.height
        if count == 1:
            band = cp.asarray(dst.read(1), dtype=cp.uint8)
            nodata = dst.nodata
            con = (band != nodata)
            return band, con
        else:
            band_list = []
            con = None
            nodata = dst.nodata
            for idx in range(count):
                band = cp.asarray(dst.read(idx+1), dtype=cp.float16)
                band_list.append(band)
                if idx == 0:
                    con = (band != nodata)
                else:
                    con &= (band != nodata)
            band = cp.stack(band_list, axis=2)
            return band, con


def change_effect_of_inertia(cur_diff_list, pre_diff_list, accelerate_rate=0.1):
    '''
    Changing Inertia Coefficient of Each Land Use Type

    Parameters:
    ---------------------
    cur_diff_list: list
        goal - cur
    pre_diff_list: list
        goal - pre
    accelerate_rate: float
        accelerate_rate
    '''

    for idx in range(len(cur_diff_list)):
        cur_diff = float(cur_diff_list[idx])
        pre_diff = float(pre_diff_list[idx])

        if abs(cur_diff) < abs(pre_diff):
            adj_rate = cur_diff / pre_diff + accelerate_rate
            if adj_rate > 1:
                if cur_diff < 0:

                    EFFECT_OF_INERTIA[idx] /= adj_rate
                elif cur_diff > 0:

                    EFFECT_OF_INERTIA[idx] *= adj_rate

        elif cur_diff < pre_diff < 0:
            EFFECT_OF_INERTIA[idx] *= (pre_diff / cur_diff)
        elif cur_diff > pre_diff > 0:
            EFFECT_OF_INERTIA[idx] *= (cur_diff / pre_diff)



def count_land_use(img, landuse_count=4):
    '''
    Return the number of each land use type

    Parameters:
    ------------------
    img: ndarray
        land use image
    landuse_count: int
        number of land use types
    '''

    land_use_count_list = []
    for idx in range(landuse_count):
        landuse = idx + 1
        land_use_count_list.append(int(cp.count_nonzero(img == landuse)))
    return cp.array(land_use_count_list)


CHANGE_COST = cp.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 0, 0, 1],
], dtype=cp.float16)

EFFECT_OF_INERTIA = cp.array([1, 1, 1, 1], dtype=cp.float16)


def get_change_cost_and_inertia_mtx(img_begin):
    
    img_cost_all = cp.zeros((img_begin.shape[0], img_begin.shape[1], 4), dtype=cp.float16)
    for from_land_idx in range(4): # 
        con = (img_begin==(from_land_idx+1))
        to_cost_list = CHANGE_COST[from_land_idx]
        img_cost_list = []
        for to_idx, to_cost in enumerate(to_cost_list):
            img_cost = cp.zeros((img_begin.shape[0], img_begin.shape[1]), dtype=cp.float16)
            img_cost[con] = to_cost
            if from_land_idx == to_idx:
                img_cost[con] *= EFFECT_OF_INERTIA[from_land_idx]
            img_cost_list.append(img_cost)
        img_cost = cp.stack(img_cost_list, axis=2)
        img_cost_all += img_cost


    return img_cost_all







def write_img(img, out_filename, dst_filename):
    '''
    write image

    img: ndarray
        land use array
    filename: str
        file name of the image
    dst_filename: str
        file name of the reference image(in order to obtain file format information)
    '''
    with rasterio.open(dst_filename) as dst:
        profile = dst.profile
    


    if rasterio.shutil.exists(out_filename):
        rasterio.shutil.delete(out_filename)

    with rasterio.open(out_filename, 'w', **profile) as dst:
        dst.write(img, 1)



def main(begin_img_path='../data/2000.tif', p_path='../data/p.tif', number_of_iter=50, goal_list=cp.array([1340450, 2976920, 96875, 420815, 609349, 1886]), out_filename='../output/re3.tif', dst_filename='../data/2000.tif', 
        neighrbour_radius=5):
    
    global EFFECT_OF_INERTIA
    EFFECT_OF_INERTIA = cp.array([1, 1, 1, 1], dtype=cp.float16)

    
    img_begin, con1 = read_img(begin_img_path)
    
    img_possibility_of_occurence, con2 = read_img(p_path)
    
    

    valid_con = con1 & con2

    del con1, con2


    land_use_count_begin = count_land_use(img_begin)
    land_use_count_cur = land_use_count_begin.copy()
    land_use_count_pre = land_use_count_begin.copy()

    land_use_count_goal = goal_list


    ca_tick = time()


    n_time_all = 0
    c_time_all = 0
    r_time_all = 0
    iter_cnt = 0
    begin_tick = time()
    for iter_idx in range(number_of_iter):
        
        # mempool.free_all_blocks()
        # pinned_mempool.free_all_blocks()

        land_use_count_cur = count_land_use(img_begin) # the current number of pixels for each land use type

        cur_diff_list = land_use_count_goal - land_use_count_cur # t-1 diff
        pre_diff_list = land_use_count_goal - land_use_count_pre # t-2 diff
        change_effect_of_inertia(cur_diff_list, pre_diff_list)
        land_use_count_pre = land_use_count_cur.copy()


        print('{0}: '.format(iter_idx), land_use_count_cur) # 

        if cp.all(land_use_count_cur == land_use_count_goal):
            break
        if cp.sum(cp.abs((land_use_count_cur - land_use_count_pre))) == 0 and iter_idx > 50:
            break


        img_tmp = img_begin.copy()
        
        tick = time()
        # Neighborhood Probability
        img_neighbour = count_neighbour_vectorize(img_begin, moor_radius=neighrbour_radius)
        n_time = time() - tick
        n_time_all += n_time
        print('neighbour-', n_time)
        # Conversion Cost and Inertia

        tick = time()
        img_cost_inertia = get_change_cost_and_inertia_mtx(img_begin)
        c_time = time() - tick
        c_time_all += c_time
        print('cost-', c_time)

        tick = time()


        possibility_of_change_mtx = img_neighbour * img_possibility_of_occurence * img_cost_inertia

        possibility_of_change_mtx = possibility_of_change_mtx / cp.sum(possibility_of_change_mtx, axis=2, keepdims=True)
        
        roulette_r = (cp.cumsum(possibility_of_change_mtx, axis=2) > cp.random.uniform(size=(possibility_of_change_mtx.shape[0], possibility_of_change_mtx.shape[1], 1), dtype=cp.float32)).astype(cp.int8) # 获得轮盘结果
        roulette_r = (4 - cp.sum(roulette_r, axis=2)) + 1

        r_time = time() - tick
        r_time_all += r_time
        print('roulette-', r_time)

        # save change
        land_type_list_1 = cp.arange(1, 5)
        cp.random.shuffle(land_type_list_1)

        land_type_list_2 = cp.arange(1, 5)
        cp.random.shuffle(land_type_list_2)

        for old_land_type in land_type_list_1:
            for new_land_type in land_type_list_2:

                if old_land_type == new_land_type:
                    continue

                old_begin_count = land_use_count_begin[old_land_type-1]
                old_goal_count = land_use_count_goal[old_land_type - 1]
                old_cur_count = land_use_count_cur[old_land_type-1]

                new_begin_count = land_use_count_begin[new_land_type-1]
                new_goal_count = land_use_count_goal[new_land_type - 1]
                new_cur_count = land_use_count_cur[new_land_type-1]

                
                if iter_idx > 10:                
                    if old_cur_count <= old_goal_count:
                        continue
                    if new_cur_count >= new_goal_count:
                        continue



                if (old_begin_count >= old_goal_count) and (old_cur_count <= old_goal_count):
                    EFFECT_OF_INERTIA[old_land_type-1] = 1
                    continue
                if (new_begin_count <= new_goal_count) and (new_cur_count >= new_goal_count):
                    EFFECT_OF_INERTIA[new_land_type-1] = 1
                    continue

                con4 = img_possibility_of_occurence[:,:,new_land_type-1] > ((cp.random.rand() + 1/4.0 + 0.1) / (iter_idx + 1)) # 
                con5 = roulette_r == new_land_type # 
                con6 = img_begin == old_land_type # 

                con_all = valid_con & con4 & con5 & con6
                del con4, con5, con6

                new_p_list = img_possibility_of_occurence[:,:,new_land_type-1][con_all]

                limit_cnt = 10000
                if new_p_list.size > limit_cnt:
                    new_p_list.astype(cp.float32).sort()
                    con7 = img_possibility_of_occurence[:,:,new_land_type-1] > new_p_list[limit_cnt-1]
                    con_all = con_all & con7


                count_of_change = cp.count_nonzero(con_all)

                old_tmp_count = old_cur_count - count_of_change # 
                new_tmp_count = new_cur_count + count_of_change # 

                if ((old_begin_count >= old_goal_count) and (old_tmp_count < old_goal_count)) or \
                    ((new_begin_count <= new_goal_count) and (new_tmp_count > new_goal_count)):
                    diff_old = old_cur_count - old_goal_count
                    diff_new = new_goal_count - new_cur_count



                    count_of_change_adj = min(diff_old, diff_new)
                    diff_count_of_change = count_of_change - count_of_change_adj

                    row_list, col_list = cp.where(con_all)
                    randomize = cp.arange(len(row_list))
                    cp.random.shuffle(randomize)
                    randomize = randomize[:diff_count_of_change]
                    row_list_adj = row_list[randomize]
                    col_list_adj = col_list[randomize]

                    con_all[row_list_adj,col_list_adj] = False

                adj_count_of_change = cp.count_nonzero(con_all)

                land_use_count_cur[old_land_type-1] -= adj_count_of_change
                land_use_count_cur[new_land_type-1] += adj_count_of_change

                img_tmp[con_all] = new_land_type


        img_begin = img_tmp

        del roulette_r, img_neighbour, img_cost_inertia, possibility_of_change_mtx
        # print('ca time', time() - ca_tick)
        iter_cnt+=1


    

    # print('n_time: ', n_time_all)
    # print('c_time: ', c_time_all)
    # print('r_time: ', r_time_all)

    write_img(cp.asnumpy(img_begin), out_filename, dst_filename)

if __name__ == '__main__':

    main(
        begin_img_path='./data/2000_4cls.tif',
        p_path='./data/p.tif',
        number_of_iter=1,
        goal_list=cp.array([13837548,33352983,4463947,8207920]),
        out_filename='./output/result_{}.tif'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S')),
        dst_filename='./data/2000_4cls.tif',
        neighrbour_radius=3)

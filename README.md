# Tensor-CA  : A high-performance cellular automata model for multiple land use simulation based on vectorization and GPU

## Data
As the data cannot be accommodated in github.
It is stored separately on Baidu Wangpan and can be downloaded from https://pan.baidu.com/s/1DTmc9YxXEVng-GVVS3yQlA.
The password is `s7xn` 


## usage

Unzip the downloaded data_of_resolution.zip file, and then modify the arguments 
of main function in `tensor_ca.py`.

* begin_image_path: land use map of the initial year.
* p_path: development suitability map
* number_of_iter: rounds of simulation
* goal_list: target land use demand
* out_filename: save path for simulation result
* dst_filename: same as begin_img_path for save tiff
* neighbour_radius: neighbour radius of CA

```
main(
    begin_img_path='./data_of_resolution/data_raw_30/2000_30.tif',
    p_path='./data_of_resolution/data_raw_30/p_30.tif',
    number_of_iter=10,
    goal_list=cp.array([13837548,33352983,4463947,8207920]),
    out_filename='./result_{}.tif'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S')),
    dst_filename='./data_of_resolution/data_raw_30/2000_30.tif',
    neighbour_radius=3)
```

### run tensor-ca

```
python tensor_ca.py

```
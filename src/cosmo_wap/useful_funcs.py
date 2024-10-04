import numpy as np


def get_avg_dist(obs_pos):
    """create x_c to integrate over - lets say it's a 1000 MPc/h (128x128x128) grid situated at (x,y,z)"""
    Nside_theory= 128
    conf_space = np.linspace(0,1000,Nside_theory)
    x_unorm , y_unorm , z_unorm = np.meshgrid(conf_space-obs_pos[0], conf_space-obs_pos[1], conf_space-obs_pos[2],indexing='ij') 
    conf_norm = np.sqrt(x_unorm**2 + y_unorm**2 + z_unorm**2) # make a unit vector - normalise
    ds = np.where(conf_norm==0,1,conf_norm)
    return 1/np.mean(1/ds**3)


def index_tuple(tup,index_cmd):
    """index the array in a tuple"""
    indexed_list = []
    for i,item in enumerate(tup):
        if np.array(item).size>1:
            indexed_list.append(item[index_cmd])
        else:
            indexed_list.append(item)
    return indexed_list  


def tuple_bool(tup,bool_arr):
    """boolean the array in a tuple - when the array are n-dimensional with last two dimensions to 1 """
    indexed_list = []
    for i,item in enumerate(tup):
        if np.array(item).size>1:

            indexed_list.append(item.flatten()[bool_arr][...,None,None])
        else:
            indexed_list.append(item)
    return indexed_list

def get_args(func,params,derivs,betas,r,s,paramsPNG = []):
    """
    get appropiate args for each function type
    """
    func_id = func.__name__ # get string name of func

    args = [params]
    if 'PNG' in func_id:
        args.append(paramsPNG)
    if 'RR' in func_id:
        args.append(derivs)
    if 'GR' in func_id:
        args.append(betas)
    if 'RR' in func_id or 'WA' in func_id:
        args.append(r)
        args.append(s)
    return args
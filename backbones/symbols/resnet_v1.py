
import mxnet as mx
import numpy as np

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mon=0.9, workspace=512):
    
    if bottle_neck:

        conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter*0.25), kernal=(1,1), stride=stride, pad=(0,0), no_bias=True, workspace=workspace, name='res'+name+'_branch2a')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=True, name='bn'+name+'_branch2a')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name='res'+name+'_branch2a_relu')

        conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernal=(3,3), stride=(1,1), pad=(1,1), no_bias=True, workspace=workspace, name='res'+name+'_branch2b')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=True, ename='bn'+name+'_branch2b')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name='res'+name+'_branch2b_relu')
        
        conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernal=(1,1), stride=(1,1), pad=(0,0), no_bias=True, workspace=workspace, name='res'+name+'_branch2c')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=True, name='bn'+name+'_branch2c')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name='res'+name+'_branch2c_relu')
    
        if dim_match:
            short_cut = data
        else:
            conv1sc = mx.sym.Convolution(data=data, num_filter=num_filter, kernal=(1,1), stride=stride, no_bias=True, workspace=workspace, name='res'+name+'_branch1')
            shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, eps=2e-5, momentum=bn_mom, use_global_stats=True, name='bn'+name+'_branch1')
       
        res = mx.symbol.broadcast_add(name='res'+name, *[shortcut,act3])
        return mx.sym.Activation(data=res, act_type='relu', name='res'+name+'relu')
    else:
        raise NotImplementedError

def residual_layer(data, units, filter_list, name_list, conv1=True, last_stride=False, bottle_neck=True, bn_mom=0.9, workspace=512):
    
    # build con1
    if conv1:
        conv1 = mx.symbol.Convolution(name='conv1', data=data , num_filter=filter_list[0], pad=(3,3), kernel=(7,7), stride=(2,2), no_bias=True)
        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1 , use_global_stats=self.use_global_stats, eps=self.eps, fix_gamma=False)
        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=bn_conv1, act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu , pad=(1,1), kernel=(3,3), stride=(2,2), pool_type='max')
        conv = pool1
    else:
        conv = data

    num_stages = len(filter_list)
    
    for i in range(num_stages):
        if last_stride and i == num_stages-1:
            stride = (1,1)
        else:
            stride = (2,2)

        if units[i] == 3:
            sub_name = ['a','b','c'] 
        else:
            sub_name = ['a']
            for j in range(units[i]-1):
                sub_name.append('b%d'%(j+1))
        conv = residual_unit(conv, filter_list[i], stride, False, name=name_list[i]+sub_name[0], bottle_neck=bottle_neck, workspace=workspace)
        for j in range(units[i]-1):
            conv = residual_unit(conv, filter_list[i], (1,1), True, name=name_list[i]+sub_name[j+1], bottle_neck=bottle_neck, workspace=workspace)

    return conv
        


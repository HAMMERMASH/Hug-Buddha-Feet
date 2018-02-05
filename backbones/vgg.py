import mxnet as mx

def vgg_block(net, num_filter, num_convs, name, do_pool=True, do_batchnorm=False, workspace=512):
    
    for i in range(num_convs):
        net = mx.sym.Convolution(data=net, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), name=name+'_{}'.format(i+1))
        if do_batchnorm: 
            net = mx.sym.BatchNorm(data=net, name='bn_'+name[-1]+'_{}'.format(i+1))
        net = mx.sym.Activation(data=net, act_type='relu', name=name+'_{}_relu'.format(i+1))
    if do_pool:
        net = mx.sym.Pooling(data=net, pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='max', name=name+'_pool')
    return net 

def vgg_layer(net, filter_list = [64,128,256,512,512], num_convs=[2,2,3,3,3], name_list=['1','2','3','4','5'], last_pool=True):
    
    num_stages = len(filter_list)
    for i in range(num_stages):
        do_pool = True
        if not last_pool and i == num_stages-1:
            do_pool = False
        net = vgg_block(net, filter_list[i], num_convs[i], 'conv'+name_list[i], do_pool)

    return net 
        

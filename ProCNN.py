import theano
from theano import tensor as T
import numpy as np

#################################### Helper methods ####################################


def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 


#################################### setup ####################################

# initialize dims
v_size = 5
b_0_size = 1000
b_1_size = 1000
b_2_size = 1

rng = np.random.RandomState(3435)

W_0 = theano.shared(rng.randn(v_size,b_0_size), name='W_0') 
b_0 = theano.shared(rng.randn(b_0_size,), name='b_0') # 

W_1 = theano.shared(rng.randn(b_0_size,b_1_size), name='W_1') 
b_1 = theano.shared(rng.randn(b_1_size,), name='b_1') 

W_2 = theano.shared(rng.randn(b_1_size,b_2_size), name='W_2') 
b_2 = theano.shared(rng.randn(b_2_size,), name='b_2') 

input_vec = T.vector('input_vec') # (5,1)

layer0_output = T.nnet.sigmoid(T.dot(input_vec, W_0) + b_0)
layer1_output = T.nnet.sigmoid(theano.dot(layer0_output, W_1) + b_1)
layer2_output = T.nnet.sigmoid(theano.dot(layer1_output, W_2) + b_2)

#values, updates = theano.scan(OneStep, outputs_info=input, n_steps=10)
MLP = theano.function([input_vec], layer2_output)

print(MLP(np.asarray([1,0,1,1,0])))


#################################### training ####################################
y = T.ivector('y')

# cost function

# 



#################################### prediction ####################################






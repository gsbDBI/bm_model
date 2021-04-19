"""
Pyro models for person-item response

Author: Zhaolei (Henry) Shi -- zshi2@stanford.edu
"""

from pdb import set_trace

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim

def sigmoid_torch(z):
    return (1. / (1. + torch.exp(- z)))

"""
 The 2 parameter traditional IRT model
"""

def pred_fn_2param(theta, alpha, beta, stu_id, ques_id):
    theta_i = theta[stu_id]
    alpha_j = alpha[ques_id]
    beta_j = beta[ques_id]
    prob = sigmoid_torch(alpha_j * (theta_i - beta_j))
    return prob

def model_2param(ques_id, stu_id, correct, n_ques = None, n_stu = None, subsample_size = None):
    if not n_ques: n_ques = torch.unique(ques_id).size()[0]
    if not n_stu: n_stu = torch.unique(stu_id).size()[0]

    # link params to models defined in model function
    theta = pyro.sample("theta", dist.Normal(torch.zeros(n_stu), torch.ones(n_stu)).independent(1))
    alpha = pyro.sample("alpha", dist.Normal(torch.zeros(n_ques), torch.ones(n_ques)).independent(1))
    beta = pyro.sample("beta", dist.Normal(torch.zeros(n_ques), torch.ones(n_ques)).independent(1))

    # actual parameters corresponding to data
    prob = pred_fn_2param(theta, alpha, beta, stu_id, ques_id)

    if subsample_size:
        with pyro.plate("data", size = len(correct), subsample_size = subsample_size) as ind:
            pyro.sample("obs", dist.Bernoulli(prob[ind]), obs=correct[ind])
    else:
        with pyro.plate("data", len(correct)):
            pyro.sample("obs", dist.Bernoulli(prob), obs=correct)

def guide_2param(ques_id, stu_id, correct, n_ques = None, n_stu = None, subsample_size = None):
    if not n_ques: n_ques = torch.unique(ques_id).size()[0]
    if not n_stu: n_stu = torch.unique(stu_id).size()[0]

    theta_loc = pyro.param('theta_loc', torch.randn(n_stu, dtype = torch.float64) / 100.)
    theta_scale = pyro.param('theta_scale', torch.ones(n_stu, dtype = torch.float64),
                               constraint=constraints.positive)

    alpha_loc = pyro.param('alpha_loc', torch.ones(n_ques, dtype = torch.float64),
                               constraint=constraints.positive)
    alpha_scale = pyro.param('alpha_scale', torch.ones(n_ques, dtype = torch.float64),
                               constraint=constraints.positive)

    beta_loc = pyro.param('beta_loc', torch.randn(n_ques, dtype = torch.float64) / 100.)
    beta_scale = pyro.param('beta_scale', torch.ones(n_ques, dtype = torch.float64),
                               constraint=constraints.positive)

    # link params to models defined in model function
    pyro.sample("theta", dist.Normal(theta_loc, theta_scale).independent(1))
    pyro.sample("alpha", dist.Normal(alpha_loc, alpha_scale).independent(1))
    pyro.sample("beta", dist.Normal(beta_loc, beta_scale).independent(1))


"""
Factorization model
"""

def pred_fn_factorization(theta, alpha, beta, stu_id, ques_id):
    theta_i = theta[stu_id, :]
    alpha_j = alpha[ques_id, :]
    beta_j = beta[ques_id]
    prob = sigmoid_torch(torch.sum(alpha_j * theta_i, dim = 1) - beta_j)
    return prob

def model_factorization(ques_id, stu_id, correct, n_latent, n_ques = None, n_stu = None):
    if not n_ques: n_ques = torch.unique(ques_id).size()[0]
    if not n_stu: n_stu = torch.unique(stu_id).size()[0]

    # link params to models defined in model function
    theta = pyro.sample("theta", dist.Normal(torch.zeros(n_stu, n_latent), torch.ones(n_stu, n_latent)).independent(2))
    alpha = pyro.sample("alpha", dist.Normal(torch.zeros(n_ques, n_latent), torch.ones(n_ques, n_latent)).independent(2))
    beta = pyro.sample("beta", dist.Normal(torch.zeros(n_ques), torch.ones(n_ques)).independent(1))

    # actual parameters corresponding to data
    prob = pred_fn_factorization(theta, alpha, beta, stu_id, ques_id)

    with pyro.plate("data", len(correct)):
        pyro.sample("obs", dist.Bernoulli(prob), obs=correct)

def guide_factorization(ques_id, stu_id, correct, n_latent, n_ques = None, n_stu = None):
    if not n_ques: n_ques = torch.unique(ques_id).size()[0]
    if not n_stu: n_stu = torch.unique(stu_id).size()[0]

    theta_loc = pyro.param('theta_loc', torch.randn(n_stu, n_latent, dtype = torch.float64) / 100.)
    theta_scale = pyro.param('theta_scale', torch.ones(n_stu, n_latent, dtype = torch.float64),
                               constraint=constraints.positive)

    alpha_loc = pyro.param('alpha_loc', torch.ones(n_ques, n_latent, dtype = torch.float64),
                               constraint=constraints.positive)
    alpha_scale = pyro.param('alpha_scale', torch.ones(n_ques, n_latent, dtype = torch.float64),
                               constraint=constraints.positive)

    beta_loc = pyro.param('beta_loc', torch.randn(n_ques, dtype = torch.float64) / 100.)
    beta_scale = pyro.param('beta_scale', torch.ones(n_ques, dtype = torch.float64),
                               constraint=constraints.positive)

    # link params to models defined in model function
    pyro.sample("theta", dist.Normal(theta_loc, theta_scale).independent(2))
    pyro.sample("alpha", dist.Normal(alpha_loc, alpha_scale).independent(2))
    pyro.sample("beta", dist.Normal(beta_loc, beta_scale).independent(1))


"""
Hierarchical factorization model (using attributes)
"""

def pred_fn_hierarchical(theta, alpha, beta, ques_hmat_list, stu_id, ques_id, ques_attrib_list):
    hier = []
    for ques_hmat, ques_attrib in zip(ques_hmat_list, ques_attrib_list):
        # ques_attrib are composed of n_obs * n_var matrices
        # ques_hmat_list are composed of n_var * ques_n_latent matrices
        hier.append(torch.matmul(ques_attrib, ques_hmat))
    theta_i = theta[stu_id, :]
    alpha_j = alpha[ques_id, :]
    hier_j = [x[ques_id, :] for x in hier]
    alphahier_j = torch.cat([alpha_j] + hier_j, dim = 1)
    beta_j = beta[ques_id]
    prob = sigmoid_torch(torch.sum(alphahier_j * theta_i, dim = 1) - beta_j)
    return prob

def model_hierarchical(ques_id, stu_id, correct, n_latent, ques_attrib_list, ques_attrib_n_latent, n_ques = None, n_stu = None):
    if not n_ques: n_ques = torch.unique(ques_id).size()[0]
    if not n_stu: n_stu = torch.unique(stu_id).size()[0]

    # attribute and trasformation matrices
    ques_hmat_list = []
    counter = 0
    total_ques_n_latent = 0
    for ques_attrib, ques_n_latent in zip(ques_attrib_list, ques_attrib_n_latent):
        n_var = ques_attrib.shape[1]
        ques_hmat_list.append(
            pyro.sample(
                "ques_hmat_{}".format(counter),
                dist.Normal(torch.zeros(n_var, ques_n_latent), torch.ones(n_var, ques_n_latent)).independent(2)
            ))
        counter += 1
        total_ques_n_latent += ques_n_latent

    # latent parameters
    theta = pyro.sample("theta", dist.Normal(torch.zeros(n_stu, n_latent + total_ques_n_latent), torch.ones(n_stu, n_latent + total_ques_n_latent)).independent(2))
    alpha = pyro.sample("alpha", dist.Normal(torch.zeros(n_ques, n_latent), torch.ones(n_ques, n_latent)).independent(2))
    beta = pyro.sample("beta", dist.Normal(torch.zeros(n_ques), torch.ones(n_ques)).independent(1))

    # actual parameters corresponding to data
    prob = pred_fn_hierarchical(
            theta = theta, alpha = alpha, beta = beta,
            ques_hmat_list = ques_hmat_list, stu_id = stu_id,
            ques_id = ques_id, ques_attrib_list = ques_attrib_list
            )

    with pyro.plate("data", len(correct)):
        pyro.sample("obs", dist.Bernoulli(prob), obs=correct)

def guide_hierarchical(ques_id, stu_id, correct, n_latent, ques_attrib, ques_attrib_n_latent, n_ques = None, n_stu = None):
    if not n_ques: n_ques = torch.unique(ques_id).size()[0]
    if not n_stu: n_stu = torch.unique(stu_id).size()[0]

    ques_hmat_guides = {}
    counter = 0
    total_ques_n_latent = 0
    for ques_attrib, ques_n_latent in zip(ques_attrib, ques_attrib_n_latent):
        n_var = ques_attrib.shape[1]
        hmat_name = "ques_hmat_{}".format(counter)
        ques_hmat_guides[hmat_name] = {
            'loc' :  pyro.param(
                        '{}_loc'.format(hmat_name),
                        torch.ones(n_var, ques_n_latent, dtype = torch.float64),
                        constraint=constraints.positive
                    ),
            'scale' :  pyro.param(
                        '{}_scale'.format(hmat_name),
                        torch.ones(n_var, ques_n_latent, dtype = torch.float64),
                        constraint=constraints.positive
                    )
        }
        counter += 1
        total_ques_n_latent += ques_n_latent


    theta_loc = pyro.param('theta_loc', torch.randn(n_stu, n_latent + total_ques_n_latent, dtype = torch.float64) / 100.)
    theta_scale = pyro.param('theta_scale', torch.ones(n_stu, n_latent + total_ques_n_latent, dtype = torch.float64),
                               constraint=constraints.positive)

    alpha_loc = pyro.param('alpha_loc', torch.ones(n_ques, n_latent, dtype = torch.float64),
                               constraint=constraints.positive)
    alpha_scale = pyro.param('alpha_scale', torch.ones(n_ques, n_latent, dtype = torch.float64),
                               constraint=constraints.positive)

    beta_loc = pyro.param('beta_loc', torch.randn(n_ques, dtype = torch.float64) / 100.)
    beta_scale = pyro.param('beta_scale', torch.ones(n_ques, dtype = torch.float64),
                               constraint=constraints.positive)

    # link params to models defined in model function
    for hmat_name in ques_hmat_guides:
        pyro.sample(
            hmat_name,
            dist.Normal(
                ques_hmat_guides[hmat_name]['loc'],
                ques_hmat_guides[hmat_name]['scale']
            ).independent(2)
        )
    pyro.sample("theta", dist.Normal(theta_loc, theta_scale).independent(2))
    pyro.sample("alpha", dist.Normal(alpha_loc, alpha_scale).independent(2))
    pyro.sample("beta", dist.Normal(beta_loc, beta_scale).independent(1))

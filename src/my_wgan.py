import torch
import torch.nn
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import newmodels
import datagen
import data_manage as mydata
import torch.utils.data as data
import copy 

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
# train = mydata.
mb_size = 8
# z_dim = 10
# X_dim = 256
# y_dim = 512
# h_dim = 128
cnt = 0
lr = 1e-4


# G = torch.nn.Sequential(
#     torch.nn.Linear(z_dim, h_dim),
#     torch.nn.ReLU(),
#     torch.nn.Linear(h_dim, X_dim),
#     torch.nn.Sigmoid()
# )

G = newmodels.MultiFilterUp(8, 1, 4, use_bias = False)


D = newmodels.Critic(1, 4, 128)

A = newmodels.MultiFilter(1, 1, 4, 4, last_act = torch.nn.Sigmoid, use_bias = False, use_dropout = False)

def reset_grad():
    G.zero_grad()
    D.zero_grad()


G_solver = optim.RMSprop(A.parameters(), lr=0.001)
D_solver = optim.RMSprop(D.parameters(), lr=lr)

if __name__ == '__main__':
    data_dir = os.path.join('cloud_data', 'vline', 'val')
    net_input_name = 'polar_partial2d_q1'
    target_name = 'polar_full2d_q1'
    data_list = os.listdir(data_dir)[0: 8]
    mydataset = mydata.PointDataSet(data_dir, data_list, net_input_name, target_name, pre_processed = True)
    mydataloader = data.DataLoader(mydataset, batch_size = mb_size, shuffle= False, num_workers= 4)
    # myiter = iter(mydataloader)
    # z = Variable(torch.randn(mb_size, 8, 8, 16))
    # y = G(z)
    # print(y.size())
    num_iter = 100
    n_critic_iter = 10
    curr_statesA = copy.deepcopy(A.state_dict())
    curr_statesD = copy.deepcopy(D.state_dict())
    for it in range(num_iter):
        # for _ in range(n_critic_iter):
        # train critic
        A.eval()
        D.train()
        for batch_idx, sample in enumerate(mydataloader):
        # Sample data
            # z = Variable(torch.randn(mb_size, 8, 8, 16))
            # sample = myiter.next()
            # X = Variable(sample[target_name])
            X = sample[target_name]
            # X = Variable(torch.from_numpy(X))

            # Dicriminator forward-loss-backward-update
            # G_sample = A(X)#G(z)
            D_real = D(X)
            D_fake = D(A(X))

            D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

            D_loss.backward()
            D_solver.step()

            # Weight clipping
            # for p in D.parameters():
            #     p.data.clamp_(-0.01, 0.01)
            reset_grad()

        # TODO make sure G params not changed here
        # check generator A did not change
        # everything_is_good = True
        # new_gradients = [param.grad for param in A.parameters()]
        # for grad in new_gradients:
        #     if grad == None:
        #         print(e)
        #         raise ValueError('found some grad none')
        #         # everything_is_good = False

        # new_states = copy.deepcopy(A.state_dict())
        # for state_name in new_states:
        #     old_state = curr_statesA[state_name]
        #     new_state = new_states[state_name]
        #     if not torch.equal(old_state, new_state):
        #         # print('state_name, 'not changed'')
        #         everything_is_good = False
        # if not everything_is_good:
        #     print('A state changed during training for D')
        # print('everything is good?', everything_is_good)
        # curr_statesA = new_states

        # curr_statesD =  copy.deepcopy(D.state_dict())
            # Housekeeping - reset gradient
        # Generator forward-loss-backward-update
        # X, _ = mnist.train.next_batch(mb_size)
        # X = Variable(torch.from_numpy(X))
        # sample = myiter.next()
        A.train()
        D.eval()
        for batch_idx, sample in enumerate(mydataloader):
            X = sample[target_name]
            z = Variable(torch.randn(mb_size, 8, 8, 16))

            G_sample = A(X)# G(z)
            # print(G_sample.size())
            D_fake = D(G_sample)

            G_loss = -torch.mean(D_fake)

            G_loss.backward()
            G_solver.step()

            # Housekeeping - reset gradient
            reset_grad()
        # check generator A did not change
        # everything_is_good = True
        # new_gradients = [param.grad for param in D.parameters()]
        # for grad in new_gradients:
        #     if grad == None:
        #         print(e)
        #         raise ValueError('found some grad none')
        #         # everything_is_good = False

        # new_states = copy.deepcopy(D.state_dict())
        # for state_name in new_states:
        #     old_state = curr_statesD[state_name]
        #     new_state = new_states[state_name]
        #     if not torch.equal(old_state, new_state):
        #         # print('state_name, 'not changed'')
        #         everything_is_good = False
        # if not everything_is_good:
        #     print('D state changed during training for A')
        # curr_statesD = new_states
        # curr_statesA = copy.deepcopy(A.state_dict())


        # Print and plot every now and then
        if it % 10 == 0:
            print('Iter-{}; D_loss: {}; G_loss: {}'
                .format(it, D_loss.data.numpy(), G_loss.data.numpy()))

            # samples = G(z).data.numpy()[:16]
            G_np = G_sample.detach().numpy()
            plt.figure()
            plt.imshow(G_np[0, 0, :, :], cmap = 'gray')
            plt.title('output')
            plt.show()
            X_np = X.detach().numpy()
            plt.figure()
            plt.imshow(X_np[0, 0, :, :], cmap = 'gray')
            plt.title('target')
            plt.show()
            # fig = plt.figure(figsize=(4, 4))
            # gs = gridspec.GridSpec(4, 4)
            # gs.update(wspace=0.05, hspace=0.05)

            # for i, sample in enumerate(samples):
            #     ax = plt.subplot(gs[i])
            #     plt.axis('off')
            #     ax.set_xticklabels([])
            #     ax.set_yticklabels([])
            #     ax.set_aspect('equal')
            #     plt.imshow(sample.reshape(256, 512), cmap='Greys_r')

            # if not os.path.exists('out/'):
            #     os.makedirs('out/')

            # plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
            cnt += 1
            # plt.close(fig)
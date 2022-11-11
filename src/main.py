import argparse
import numpy as np
from data_loader import load_data
from train import train

np.random.seed(555)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
# 实体、关系等的embedding向量维度
parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
# 采样跳数
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
# 更新kg实体、关系的embedding向量的lambda2值大小，在损失函数中使用
parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
# 正则化因子的lambda值大小
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
# 梯度下降法，更新参数的学习率
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
# 每次随机采样数据的大小
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
# 迭代次数epoch
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
# ripple set的大小
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
# 更新每一跳采样实体的向量表示的方式
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
# 控制user的向量构成，false时使用最后一跳得到的o，true时使用全部o想加
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')

'''
# default settings for Book-Crossing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--dim', type=int, default=4, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=1e-2, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
'''

args = parser.parse_args()

show_loss = False
data_info = load_data(args)
train(args, data_info, show_loss)

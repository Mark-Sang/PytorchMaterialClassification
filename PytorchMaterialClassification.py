from dataload import Front_viewPhotos,Lateral_viewPhotos
from config import DefaultConfig
from Net import MyNet_Front,MyNet_Lateral,MyNet_Correlation,MyNet_New
from torch.autograd import Variable

import torch
import numpy as np
import csv
import fire
import visdom

#参数配置opt
opt=DefaultConfig()
vis_loss = []                           #loss值
vis_meanloss = []                       #一个循环的平均loss
vis_x = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,
                      27,28,29,30])

#加载图片
train_iron_front=Front_viewPhotos(opt.train_iron_front_root)
train_iron_lateral=Lateral_viewPhotos(opt.train_iron_lateral_root)
train_wood_front=Front_viewPhotos(opt.train_wood_front_root)
train_wood_lateral=Lateral_viewPhotos(opt.train_wood_lateral_root)

test_front=Front_viewPhotos(opt.test_front_root)
test_lateral=Lateral_viewPhotos(opt.test_lateral_root)

#加载网络
#Net_F = MyNet_Front()
#Net_L = MyNet_Lateral()
#Net_C = MyNet_Correlation()
Net_N = MyNet_New()

#优化器
#optimizer_F = torch.optim.Adam(Net_F.parameters(), lr=0.002, betas=(0.5, 0.999))
#optimizer_L = torch.optim.Adam(Net_L.parameters(), lr=0.002, betas=(0.5, 0.999))
#optimizer_C = torch.optim.Adam(Net_C.parameters(), lr=0.002, betas=(0.5, 0.999))
optimizer_N = torch.optim.Adam(Net_N.parameters(), lr=0.002, betas=(0.5, 0.999))

#损失函数
criterion = torch.nn.CrossEntropyLoss()

def train(**kwargs):
    opt.parse(kwargs)                                    #根据命令行参数配置更新
    vis = visdom.Visdom(env=opt.env)
    vis_num = 0

    if opt.use_gpu:
        Net_N.cuda()

    for k in range(8):
        for i in range(300):
            valid_iron = torch.tensor([1])
            valid_wood = torch.tensor([0])
            valid_iron = Variable(valid_iron).cuda()
            valid_wood = Variable(valid_wood).cuda()

            iron_front = train_iron_front[i].cuda()    
            iron_lateral = train_iron_lateral[i].cuda()
            iron_front = iron_front.unsqueeze(0).cuda()
            iron_lateral = iron_lateral.unsqueeze(0).cuda()

            wood_front = train_wood_front[i].cuda()
            wood_lateral = train_wood_lateral[i].cuda()
            wood_front =  wood_front.unsqueeze(0).cuda()
            wood_lateral = wood_lateral.unsqueeze(0).cuda()

            for c1 in range(3):
                optimizer_N.zero_grad()                        #第一类
                g = Net_N(iron_front, iron_lateral)
                loss_GAN = criterion(g, valid_iron)
                loss_GAN.backward()
                optimizer_N.step()

     #           print(vis_num,loss_GAN)
                vis_num = vis_num+1
                vis_loss.append(loss_GAN.item())
   #             print('\n')
            for c2 in range(3):
                optimizer_N.zero_grad()                        #第二类
                g = Net_N(wood_front, wood_lateral)
                loss_GAN = criterion(g, valid_wood)
                loss_GAN.backward()
                optimizer_N.step()

    #            print(vis_num,loss_GAN)
                vis_num = vis_num+1
                vis_loss.append(loss_GAN.item())
    #            print('\n')

        vis_loss_del = del_max_min(vis_loss)
        vis_mean = mean(vis_loss_del)
        print(k,vis_mean)
        vis_loss.clear() 
        vis_meanloss.append(vis_mean)
    #visdom显示
    vis.line(Y=vis_meanloss, 
             X=vis_x,
             opts=dict(title="600组训练集损失函数训练变化")
             )

    #save N
    torch.save(Net_N,'NetN.pkl')                             #entire net
    torch.save(Net_N.state_dict,'NetN_parames.pkl')          #parameters

def test(**kwargs):
    opt.parse(kwargs)                                     #根据命令行参数配置更新
    Net_N = torch.load('NetN.pkl')

    if opt.use_gpu:
        Net_N.cuda()
    Net_N.eval()

    for i in range(42):
        front = test_front[i].cuda()
        lateral = test_lateral[i].cuda()
        front = front.unsqueeze(0).cuda()
        lateral = lateral.unsqueeze(0).cuda()

        g = Net_N(front, lateral)
        max_value,materal_num = torch.max(g, 1)
        print(i,  materal_num)

def mean(list_vis_loss, total=0.0):                      #取平均值
    num = 0
    for i in list_vis_loss:
        total = total+i
        num = num+1
    return total/num

def del_max_min(list_vis_loss):                         #去除最大最小值
    for i in range(10):
        list_vis_loss.pop(list_vis_loss.index(max(list_vis_loss)))
        list_vis_loss.pop(list_vis_loss.index(min(list_vis_loss)))
    return list_vis_loss


if __name__ == '__main__':
    fire.Fire()
  

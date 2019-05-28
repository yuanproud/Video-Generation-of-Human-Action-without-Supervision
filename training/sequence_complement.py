# coding: utf-8

import os
import argparse
import time
import math
import skvideo.io
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable, grad
import torch.utils.data as Data
import torch.nn.functional as F
from torchvision import transforms
import imageio
import pdb
import skimage.transform
import scipy.misc
#os.environ["CDF_LIB"] = '/CDF_BASE/lib'
from spacepy import pycdf

from models import Discriminator_2d, Generator_1d, Generator_V, Encoder_1d
#from Pose_Estimation.demo.picture_demo import get_heatmap

parser = argparse.ArgumentParser(description='Start trainning MoCoGAN.....')
parser.add_argument('--cuda', type=int, default=1,
                     help='set -1 when you use cpu')
parser.add_argument('--multiple-gpu', type=int, default=-1,
                     help='set 1 when you use multiple gpu')
parser.add_argument('--ngpu', type=int, default=0,
                     help='set the number of gpu you use')
parser.add_argument('--batch-size', type=int, default=2,
                     help='set batch_size, default: 16')
parser.add_argument('--niter', type=int, default=1000,
                     help='set num of iterations, default: 120000')
parser.add_argument('--pre-train', type=int, default=-1,
                     help='set 1 when you use pre-trained models')
parser.add_argument('--image-size', type=int, default=128,
                     help='Image size')
parser.add_argument('--lambda_gp', type=float, default=10,
                     help='weight for gradient penalty')
parser.add_argument('--action', type=str, default='Walking',
                     help='action of you want to generate')

args       = parser.parse_args()
cuda       = args.cuda
multiple_gpu = args.multiple_gpu
ngpu       = args.ngpu
batch_size = args.batch_size
n_iter     = args.niter
pre_train  = args.pre_train
img_size = args.image_size
lambda_gp = args.lambda_gp
action = args.action

T = 32
nc = 3
nc_heatmap = 30
nc_center = 14
nc_limbs = 16
nc_noise = 100
LAMBDA = 10

class FaceIdPoseDataset(Data.Dataset):

    #  assume images  as B x C x H x W  numpy array
    def __init__(self, videos_ts, transform=None):

        self.videos_ts = videos_ts
        self.transform = transform

    def __len__(self):
        return len(self.videos_ts)

    def __getitem__(self, idx):

        video_ts = self.videos_ts[idx]
#        if self.transform:
#            image = self.transform(image)

        return [video_ts]


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
if cuda == True:
    torch.cuda.set_device(0)


''' prepare dataset '''
current_path = os.path.dirname(__file__)
resized_path = os.path.join(current_path, 'train_data/Pose_D2_140*100_%s'%action)

videos = []
for root, dir_names, fnames in os.walk(resized_path):
    for fname in sorted(fnames):
        path = os.path.join(root, fname)
        videos.append(path)
        
print('finish load data')
print(len(videos))
            

''' prepare video sampling '''
def Normalization(first_img):
    ''' first_img: batch_size, T, nc_heatmap '''
    bs = first_img.size(0)
    first_img = first_img.float()
    length = torch.sqrt((first_img[:,:,0] - first_img[:,:,14])**2 + (first_img[:,:,1] - first_img[:,:,15])**2) #14
    rate = (14 / length).float() # b, T
    
    for k in range(len(rate)):
        mean = torch.mean(rate[k])
        if rate[k][0] <= mean:
            for c in range(len(rate[0])-1):
                if rate[k][c+1] < rate[k][c]:
                    rate[k][c+1] = rate[k][c]
        if rate[k][0] > mean:
            for c in range(len(rate[0])-1):
                if rate[k][c+1] > rate[k][c]:
                    rate[k][c+1] = rate[k][c]
    
    traned = torch.zeros([bs, first_img.shape[1], nc_heatmap])
    for i in range(first_img.shape[1]):
        for j in range(nc_heatmap //2):
            traned[:,i,j*2] = torch.round((first_img[:,i,j*2] - 50) * rate[:,i] + 50)
            
            traned[:,i,j*2+1] = torch.round((first_img[:,i,j*2+1] - 70) * rate[:,i] + 70)
    
    return traned, rate

# for true video
def trim(video, start):
    end = start + T
    return video[:, start:end, :,], end
    
''' set models '''

criterion = nn.BCEWithLogitsLoss()
mse = nn.MSELoss()
kld = nn.KLDivLoss()
l1 = nn.L1Loss()

if multiple_gpu == True:    
    GI = nn.DataParallel(Generator_1d(nc_center, nc_heatmap))
    GV = nn.DataParallel(Generator_V(nc_noise, 2048, 100, T))
    encoder = nn.DataParallel(Encoder_1d(nc_limbs, nc_noise))
    dis_v = nn.DataParallel(Discriminator_2d(nc_noise, 1024, 1, T))
else:
    GI = Generator_1d(nc_center, nc_heatmap)
    GV = Generator_V(nc_noise, 2048, 100, T)#T*nc_noise)
    encoder = Encoder_1d(nc_limbs, nc_noise)
    dis_v = Discriminator_2d(nc_heatmap, 1024, 1, T) 
    
''' prepare for train '''
def timeSince(since):
    now = time.time()
    s = now - since
    d = math.floor(s / ((60**2)*24))
    h = math.floor(s / (60**2)) - d*24
    m = math.floor(s / 60) - h*60 - d*24*60
    s = s - m*60 - h*(60**2) - d*24*(60**2)
    return '%dd %dh %dm %ds' % (d, h, m, s)

trained_path = os.path.join(current_path, 'trained_models_%s_video'%action)
if not os.path.exists(trained_path):
    os.makedirs(trained_path)
    
dir_path = os.path.join(current_path, 'generated_videos_%s_video'%action)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    
def checkpoint(model, model_name, optimizer, epoch):
    filename = os.path.join(trained_path, model_name+'_epoch-%d' % (epoch))
    torch.save(model.state_dict(), filename + '.pkl')
    torch.save(optimizer.state_dict(), filename + '.state')

def save_video(fake_video, real_video, encoded_video, epoch):
    
    outputdata = fake_video
    outputdata = outputdata.astype(np.uint8)
    file_path = os.path.join(dir_path, 'fakeVideo_epoch-%d.mp4' % epoch)
    skvideo.io.vwrite(file_path, outputdata)
    
    realdata = real_video
    realdata = realdata.astype(np.uint8)
    file_path = os.path.join(dir_path, 'realVideo_epoch-%d.mp4' % epoch)
    skvideo.io.vwrite(file_path, realdata)
    
    realdata = encoded_video
    realdata = realdata.astype(np.uint8)
    file_path = os.path.join(dir_path, 'encodedVideo_epoch-%d.mp4' % epoch)
    skvideo.io.vwrite(file_path, realdata)
    
def calc_gradient_penalty(netD, real_data, fake_data, batch_size):
    alpha = torch.rand(real_data.size())
    #alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    # TODO: Make ConvBackward diffentiable
    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

if pre_train == True:
    GV.load_state_dict(torch.load(trained_path + '/GV_epoch-150.pkl'))
    dis_v.load_state_dict(torch.load(trained_path + '/dis_v_epoch-150.pkl'))
    
GI.load_state_dict(torch.load('trained_models_%s/unet_epoch-5000.pkl')%action)    
encoder.load_state_dict(torch.load('trained_models_%s/encoder_epoch-5000.pkl')%action)
print('Load pre-train model')

''' adjust to cuda '''

if cuda == True:
    GI.cuda()
    GV.cuda()
    encoder.cuda()
    dis_v.cuda()

# setup optimizer
lr = 0.0001
lr_dis = 0.0001
betas=(0.5, 0.999)

optim_g_v = optim.Adam(GV.parameters(), lr=lr, betas=betas)
optim_dis_v = optim.Adam(dis_v.parameters(), lr=lr_dis, betas=betas)
''' calc grad of models '''

torch_dataset = FaceIdPoseDataset(videos)
dataloader = Data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True) 
gaussian_noise = torch.distributions.normal.Normal(0,1.)
GI.eval()
encoder.eval()
''' train models '''
start_time = time.time()
l1_w = 1
for epoch in range(1, n_iter+1):
    for step, batch_data in enumerate(dataloader):
        ''' prepare real images '''
        # videos.size() => (batch_size, T, nc, img_size, img_size)
        videos_path = batch_data[0]
        mini_batch = len(videos_path)
        
        videos = []
        for i in range(mini_batch):
            cdf = pycdf.CDF(videos_path[i]) # 1, 160, 30
            videos.append(cdf['Pose'])
        videos = np.array(videos) # b, 1, 160, 30
        videos = videos.squeeze() # b, 160, 30
        start_T = np.random.randint(0, 10)
        for ind_t in range(4):
            heat_map, start_T = trim(videos, start_T)
            heat_map = torch.from_numpy(heat_map) # b, T, 30
            heat_map, zoom_rate = Normalization(heat_map)
            
            one = Variable(torch.FloatTensor(heat_map.size(0), 1).fill_(0.9), requires_grad=False).cuda()
            mone = Variable(torch.FloatTensor(heat_map.size(0), 1).fill_(0.1), requires_grad=False).cuda()
            # b, T, 13, 140, 100
            ''' prepare heat map '''
            real_image = Variable(heat_map).cuda() # b, T, 30
            real_limbs = real_image[:, :,[4,5,6,7,10,11,12,13,20,21,22,23,26,27,28,29]]
            first_real_limbs = real_image[:, 0, [4,5,6,7,10,11,12,13,20,21,22,23,26,27,28,29]]
            last_real_limbs = real_image[:, -1, [4,5,6,7,10,11,12,13,20,21,22,23,26,27,28,29]]
            real_limbs = real_limbs.view(-1, nc_limbs)
            real_limbs_encoded, _, _ = encoder(real_limbs)
            real_limbs_encoded = real_limbs_encoded.view(-1, T, nc_noise)
            first_real_limbs_encoded, _, _ = encoder(first_real_limbs) # b, nc_noise
            last_real_limbs_encoded, _, _ = encoder(last_real_limbs)
            ''' generate fake videos sequence '''
            noise = Variable(gaussian_noise.sample((mini_batch, T, nc_noise))).cuda()
            GV.initHidden(mini_batch)
            noise[:, 0, :] = first_real_limbs_encoded
            noise[:,-1, :] = last_real_limbs_encoded
            noise_T = GV(noise) # b, T*nc_noise
            noise_T = noise_T.view(mini_batch, T, nc_noise)
            ''' generate fake heat map videos '''
            real_center = real_image[:, :, [0,1,2,3,8,9,14,15,16,17,18,19,24,25]]
            fake_videos = []
            fake_videos_lat = []
            encoded_videos = []
            for t in range(T):
                fake_image = GI(real_center[:, t, :], noise_T[:, t, :])
                fake_videos.append(fake_image)
                if t>0:
                    fake_videos_lat.append(fake_image)
                if t == T-1:
                    fake_videos_lat.append(fake_image)
                    
                encoded_image = GI(real_center[:, t, :], real_limbs_encoded[:, t, :])
                encoded_videos.append(encoded_image)
            fake_videos = torch.stack(fake_videos, dim=0).permute(1,0,2)  # b, T, nc_heatmap
            fake_videos_lat = torch.stack(fake_videos_lat, dim=0).permute(1,0,2) # b, T, nc_heatmap
            encoded_videos = torch.stack(encoded_videos, dim=0).permute(1,0,2)
            #print(fake_videos)
            
            ''' train Discriminator_I '''
                
            dis_v.initHidden(mini_batch)
            dis_v.zero_grad()
            fake_loss_di = criterion(dis_v(fake_videos.detach()), mone)
            real_loss_di = criterion(dis_v(real_image), one)
            di_loss = fake_loss_di + real_loss_di
            di_loss.backward()
            optim_dis_v.step()
            
            ''' train course UNet '''
            dis_v.initHidden(mini_batch)
            GV.zero_grad()
            fake_loss_gi = criterion(dis_v(fake_videos), one)
            smooth_loss = l1(fake_videos_lat, fake_videos) / nc_heatmap
            fake_loss = fake_loss_gi + 0.001 * smooth_loss
            fake_loss.backward()
            optim_g_v.step()
            
            
            if ind_t == 0:
                fake_video = fake_videos.data.cpu()
                real_video = heat_map
                encoded_video = encoded_videos.data.cpu()
                zoom_rates = zoom_rate
            else:
                fake_video = torch.cat([fake_video, fake_videos.data.cpu()], 1)
                real_video = torch.cat([real_video, heat_map], 1)
                encoded_video = torch.cat([encoded_video, encoded_videos.data.cpu()], 1)
                zoom_rates = torch.cat([zoom_rates, zoom_rate], 1)
            
    if epoch % 5 == 0:
        print('[%d/%d] (%s) Gv_mean %.4f MSE_err %.4f, Dv_fake_mean %.4f Dv_real_mean %.4f' #Smooth_Loss %.4f
              % (epoch, n_iter, timeSince(start_time), fake_loss_gi.data, smooth_loss.data, fake_loss_di.data, real_loss_di.data)) #, smooth_loss.data
        for j in range(nc_heatmap //2):
            real_video[:,:,j*2] = torch.round((real_video[:,:,j*2] - 50)/zoom_rates) + 50
            real_video[:,:,j*2+1] = torch.round((real_video[:,:,j*2+1] - 70)/zoom_rates) + 70
            fake_video[:,:,j*2] = torch.round((fake_video[:,:,j*2] - 50)/zoom_rates) + 50
            fake_video[:,:,j*2+1] = torch.round((fake_video[:,:,j*2+1] - 70)/zoom_rates) + 70
            encoded_video[:,:,j*2] = torch.round((encoded_video[:,:,j*2] - 50)/zoom_rates) + 50
            encoded_video[:,:,j*2+1] = torch.round((encoded_video[:,:,j*2+1] - 70)/zoom_rates) + 70
        fake_video = fake_video.numpy()
        real_video = real_video.numpy()
        encoded_video = encoded_video.numpy()
        #max_fake = max(int(np.max(fake_video)), int(np.max(real_video)))
        real_all = np.zeros([T*4, 256, 256])
        fake_all = np.zeros([T*4, 256, 256])
        encoded_all = np.zeros([T*4, 256, 256])
        
        for t in range(T*4):    
            for i in range(cdf['Pose'].shape[2]//2): #4
                fake_all[t, int(fake_video[0,t, i*2+1]), int(fake_video[0, t, i*2])] = 255
                fake_all[t, int(fake_video[0,t, i*2+1])+1, int(fake_video[0,t, i*2])] = 255
                fake_all[t, int(fake_video[0,t, i*2+1])-1, int(fake_video[0,t, i*2])] = 255
                fake_all[t, int(fake_video[0,t, i*2+1]), int(fake_video[0,t, i*2])+1] = 255
                fake_all[t, int(fake_video[0,t, i*2+1]), int(fake_video[0,t, i*2])-1] = 255
                
                real_all[t, int(real_video[0,t, i*2+1]), int(real_video[0,t, i*2])] = 255
                real_all[t, int(real_video[0,t, i*2+1])+1, int(real_video[0,t, i*2])] = 255
                real_all[t, int(real_video[0,t, i*2+1])-1, int(real_video[0,t, i*2])] = 255
                real_all[t, int(real_video[0,t, i*2+1]), int(real_video[0,t, i*2])+1] = 255
                real_all[t, int(real_video[0,t, i*2+1]), int(real_video[0,t, i*2])-1] = 255
            
                encoded_all[t, int(encoded_video[0,t, i*2+1]), int(encoded_video[0,t, i*2])] = 255
                encoded_all[t, int(encoded_video[0,t, i*2+1])+1, int(encoded_video[0,t, i*2])] = 255
                encoded_all[t, int(encoded_video[0,t, i*2+1])-1, int(encoded_video[0,t, i*2])] = 255
                encoded_all[t, int(encoded_video[0,t, i*2+1]), int(encoded_video[0,t, i*2])+1] = 255
                encoded_all[t, int(encoded_video[0,t, i*2+1]), int(encoded_video[0,t, i*2])-1] = 255
                
        save_video(fake_all, real_all, encoded_all, epoch)
        
    if epoch % 80 ==0:
        l1_w *= 0.1
    
    if epoch % 80 == 0:
            
        lr *= 0.5
        lr_dis *= 0.5
        optim_g_v = optim.Adam(GV.parameters(), lr=lr, betas=betas)
        optim_dis_v = optim.Adam(dis_v.parameters(), lr=lr_dis, betas=betas)

    if epoch % 50 == 0:
        checkpoint(GV, 'GV', optim_g_v, epoch)
        checkpoint(dis_v, 'dis_v', optim_dis_v, epoch)

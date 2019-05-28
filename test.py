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
import imageio
import scipy.misc
from spacepy import pycdf

#from unet import UNet_noise
from models import Generator_1d, Generator_V, Encoder_1d
from pose_guided_models import UAE_noFC_AfterNoise, Multi_VAE, Generator
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
parser.add_argument('--niter', type=int, default=1,
                     help='set num of iterations, default: 120000')
parser.add_argument('--pre-train', type=int, default=-1,
                     help='set 1 when you use pre-trained models')
parser.add_argument('--image-size', type=int, default=256,
                     help='Image size')
parser.add_argument('--lambda_gp', type=float, default=10,
                     help='weight for gradient penalty')
parser.add_argument('--action', type=str, default='Walking',
                     help='the action you want to generate')

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

height = 256
width = 256
radius = 3
var = 4
mode = 'Solid'

class FaceIdPoseDataset(Data.Dataset):

    #  assume images  as B x C x H x W  numpy array
    def __init__(self, videos_ts, videos_cdf, videos_pl, transform=None):

        self.videos_ts = videos_ts
        self.videos_cdf = videos_cdf
        self.videos_pl = videos_pl
        self.transform = transform

    def __len__(self):
        return len(self.videos_ts)

    def __getitem__(self, idx):

        video_ts = self.videos_ts[idx]
        video_cdf = self.videos_cdf[idx]
        video_pl = self.videos_pl[idx]
#        if self.transform:
#            image = self.transform(image)

        return [video_ts, video_cdf, video_pl]
    
def _getSparseKeypoint(r, c, k, height, width, radius=4, var=4, mode='Solid'):
    r = int(r)
    c = int(c)
    k = int(k)
    indices = []
    for i in range(-radius, radius+1):
        for j in range(-radius, radius+1):
            distance = np.sqrt(float(i**2 + j**2))
            if r+i >= 0 and r+i < height and c+j >= 0 and c+j < width:
                if 'Solid' == mode and distance <= radius:
                    indices.append([r+i, c+j, k])

    return indices

def _sparse2dense(indices, shape):
    dense = np.zeros(shape)
    for i in range(len(indices)):
        r = indices[i][0]
        c = indices[i][1]
        k = indices[i][2]
        dense[r,c,k] = 1
    return dense

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
if cuda == True:
    torch.cuda.set_device(0)


''' prepare dataset '''
IMG_EXTENSIONS = [
    '.mp4', '.avi'
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

current_path = os.path.dirname(__file__)
resized_path = os.path.join(current_path, 'train_data/Pose_D2_256_%s'%action)

videos_cdf = []
for root, dir_names, fnames in os.walk(resized_path):
    for fname in sorted(fnames):
        path = os.path.join(root, fname)
        videos_cdf.append(path)

videos = []
videos_path = os.path.join(current_path, 'train_data/Videos_%s_256'%action)
for root, dir_names, fnames in os.walk(videos_path):
    for fname in sorted(fnames):
        if is_image_file(fname):
            path = os.path.join(root, fname)
            videos.append(path)
            
videos_pl = []
videos_path = os.path.join(current_path, 'train_data/Segments_mat_gt_pl_%s_256'%action)
for root, dir_names, fnames in os.walk(videos_path):
    for fname in sorted(fnames):
        if is_image_file(fname):
            path = os.path.join(root, fname)
            videos_pl.append(path)
            
print(len(videos_cdf))
print(len(videos))
print('finish load data')
            

''' prepare video sampling '''
def Normalization(first_img, real_limbs, last_limbs):
    ''' first_img: batch_size, T, nc_heatmap '''
    bs = first_img.size(0)
    first_img = first_img.float()
    length = torch.sqrt((first_img[:,:,0] - first_img[:,:,6])**2 + (first_img[:,:,1] - first_img[:,:,7])**2) #14
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
            
    
    traned = torch.zeros([bs, first_img.size(1), nc_center])
    traned_limbs = torch.zeros([bs, nc_limbs])
    traned_last_limbs = torch.zeros([bs, nc_limbs])
    for i in range(first_img.size(1)):
        for j in range(nc_center //2):
            traned[:,i,j*2] = (first_img[:,i,j*2] - 50) * rate[:,i] + 50
            traned[:,i,j*2+1] = (first_img[:,i,j*2+1] - 70) * rate[:,i] + 70
    for k in range(nc_limbs //2):
        traned_limbs[:,k*2] = (real_limbs[:,k*2] - 50) * rate[:,0] + 50
        traned_limbs[:,k*2+1] = (real_limbs[:,k*2+1] - 70) * rate[:,0] + 70
        
        traned_last_limbs[:,k*2] = (last_limbs[:,k*2] - 50) * rate[:,0] + 50
        traned_last_limbs[:,k*2+1] = (last_limbs[:,k*2+1] - 70) * rate[:,0] + 70
    
    return traned, traned_limbs, traned_last_limbs, rate

def trim(RGB, cdf, start):
    end = start + T
    return RGB[:, start:end, :, :, :], cdf[:, start:end, :], end-1
   
def interpolation(first_img, last_img, t):
    ''' first_img: batch_size, nc, img_size, img_size '''
    ''' batch_size, 30 '''
    for i in range(nc_center//2):
        arg_max_fx = first_img[:, i*2:i*2+1]
        arg_max_fy = first_img[:, i*2+1:i*2+2]
        arg_max_lx = last_img[:, i*2:i*2+1]
        arg_max_ly = last_img[:, i*2+1:i*2+2]
    
        arg_max_tx = (arg_max_fx + (arg_max_lx - arg_max_fx) / T * t).astype(float)
        arg_max_ty = (arg_max_fy + (arg_max_ly - arg_max_fy) / T * t).astype(float)
        if i == 0:
            all_t = arg_max_tx
            all_t = np.concatenate((all_t, arg_max_ty),1)
        else:
            all_t = np.concatenate((all_t, arg_max_tx), 1)
            all_t = np.concatenate((all_t, arg_max_ty), 1)
    
    return all_t # b, nc_center
''' set models '''

criterion = nn.BCELoss()
mse = nn.L1Loss()
kld = nn.KLDivLoss()

if multiple_gpu == True:    
    GI = nn.DataParallel(Generator_1d(nc_center, nc_heatmap))
    GV = nn.DataParallel(Generator_V(nc_noise, 2048, 100, T))
    encoder = nn.DataParallel(Encoder_1d(nc_limbs, nc_noise))
    unet_background = nn.DataParallel(UAE_noFC_AfterNoise(nc*2+nc_heatmap//2, 3))
    heat_to_ori = nn.DataParallel(UAE_noFC_AfterNoise(nc_heatmap//2+nc,3))
    unet_refine = nn.DataParallel(UAE_noFC_AfterNoise(nc*2, 3))
else:
    GI = Generator_1d(nc_center, nc_heatmap)
    GV = Generator_V(nc_noise, 2048, 100, T)
    encoder = Encoder_1d(nc_limbs, nc_noise)
    unet_background = UAE_noFC_AfterNoise(nc*2+nc_heatmap//2, 3)
    heat_to_ori = UAE_noFC_AfterNoise(nc_heatmap//2+nc,3)
    unet_refine = Generator()

''' prepare for train '''
def timeSince(since):
    now = time.time()
    s = now - since
    d = math.floor(s / ((60**2)*24))
    h = math.floor(s / (60**2)) - d*24
    m = math.floor(s / 60) - h*60 - d*24*60
    s = s - m*60 - h*(60**2) - d*24*(60**2)
    return '%dd %dh %dm %ds' % (d, h, m, s)

dir_path = os.path.join(current_path, 'test_videos_%s_video'%action)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def save_video(fake_video, real_video, fake_RGB_video, step): #, fake_RGB_video
    
    outputdata = fake_video
    outputdata = outputdata.astype(np.uint8)
    file_path = os.path.join(dir_path, 'fakeVideo_epoch-%d.mp4' % step)
    skvideo.io.vwrite(file_path, outputdata)
    
    realdata = real_video*255.0
    realdata = realdata.astype(np.uint8)
    file_path = os.path.join(dir_path, 'realVideo_epoch-%d.mp4' % step)
    skvideo.io.vwrite(file_path, realdata)
    
    realdata = fake_RGB_video*255.0
    realdata = realdata.astype(np.uint8)
    file_path = os.path.join(dir_path, 'fake_RGBVideo_epoch-%d.mp4' % step)
    skvideo.io.vwrite(file_path, realdata)
    for t in range(T*5):
        file_path = os.path.join(dir_path, 'fakeImage_epoch%d-T%d.jpg'%(step, t))
        scipy.misc.imsave(file_path, realdata[t].squeeze())
        
    file_path = os.path.join(dir_path, 'fake_RGB_epoch-%d.gif'%step)
    imageio.mimsave(file_path, realdata, 'GIF', duration=0.05)
    
def save_image(RGB_image, step, k):
    RGB_image = RGB_image*255.0
    RGB_image = RGB_image.astype(np.uint8)
    file_path = os.path.join(dir_path, 'conditional-%d_image_epoch-%d.png' %(k, step))
    scipy.misc.imsave(file_path, RGB_image)
    
def save_heatmap(heatmap, step, k):
    heatmap = heatmap.squeeze().astype(np.uint8)
    file_path = os.path.join(dir_path, 'conditional-%d_heatmap_epoch-%d.png' %(k, step))
    scipy.misc.imsave(file_path, heatmap)

GV.load_state_dict(torch.load('trained_models_%s_video/GV_epoch-500.pkl')%action)
GI.load_state_dict(torch.load('trained_models_%s/unet_epoch-5000.pkl')%action) 
encoder.load_state_dict(torch.load('trained_models_%s/encoder_epoch-5000.pkl')%action) 
heat_to_ori.load_state_dict(torch.load('trained_models_heatmap_to_original_SRGAN/unet_coarse_epoch-70.pkl'))   
unet_background.load_state_dict(torch.load('trained_models_heatmap_to_original_SRGAN/unet_background_epoch-70.pkl'))
unet_refine.load_state_dict(torch.load('trained_models_heatmap_to_original_SRGAN/unet_refine_epoch-70.pkl'))
#dis_i.load_state_dict(torch.load(trained_path + '/dis_i_epoch-220.pkl'))    
print('Load pre-train model')

''' adjust to cuda '''

if cuda == True:
    GI.cuda()
    GV.cuda()
    encoder.cuda()
    heat_to_ori.cuda()
    unet_background.cuda()
    unet_refine.cuda()

''' calc grad of models '''

torch_dataset = FaceIdPoseDataset(videos, videos_cdf, videos_pl)
dataloader = Data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True) 
gaussian_noise = torch.distributions.normal.Normal(0,1.)
''' train models '''
start_time = time.time()

for epoch in range(1, n_iter+1):
    for step, batch_data in enumerate(dataloader):
        ''' prepare real images '''
        # videos.size() => (batch_size, T, nc, img_size, img_size)
        videos_path = batch_data[0]
        videos_pl_path = batch_data[2]
        mini_batch = len(videos_path)
        videos_t = []
        videos_pl = []
        for k in range(mini_batch):
            vid = imageio.get_reader(videos_path[k], 'ffmpeg')
            vid_pl = imageio.get_reader(videos_pl_path[k], 'ffmpeg')
            leng = len(vid)
            al_r = []
            al_r_pl = []
            for ti in range(leng):
                image = vid.get_data(ti)
                al_r.append(image)
                image_pl = vid_pl.get_data(ti)
                al_r_pl.append(image_pl)
                
            videos_t.append(al_r)
            videos_pl.append(al_r_pl)
        videos_t = torch.FloatTensor(videos_t)
        videos_t = videos_t /255.0
        videos_t = videos_t.permute(0,1,4,2,3) # b, 160, 3, 256, 256
        
        videos_pl = torch.FloatTensor(videos_pl)
        videos_pl = videos_pl / 255.0
        videos_pl = videos_pl.permute(0,1,4,2,3)
        
        videos_cdf = batch_data[1]
        cdf_location = []
        for k in range(mini_batch):
            cdf = pycdf.CDF(videos_cdf[k]) # 1, 160, 30
            cdf_location.append(cdf['Pose'])
            
        cdf_location = np.array(cdf_location).squeeze() # b, 160, 30
        start_T = 0
        RGB_image = []
        heatmap_image = []
        fake_videos_position = []
        for ind_t in range(5):
            RGB_video, cdf_location_part, start_T = trim(videos_t, cdf_location, start_T)
            RGB_first = RGB_video[0:1,0,:,:,:].float().cuda()
            RGB_last = RGB_video[0:1,-1,:,:,:].float().cuda()
            pl_first = videos_pl[0:1,0,:,:,:].float().cuda()
            pl_last = videos_pl[0:1,-1,:,:,:].float().cuda()
            videos_forground = RGB_first * pl_first #* 2 - 1
            videos_background_f = RGB_first * torch.abs(1 - pl_first) #* 2 - 1
            videos_background_l = RGB_last * torch.abs(1 - pl_last)
            # b, T, 256, 256, 13
            ''' interpolation '''
            heatmap_first = cdf_location_part[:, 0, :] # b, 30
            heatmap_last = cdf_location_part[:, -1, :]
            interpolated_video = []
            sequence_len_x = []
            sequence_len_y = []
            first_real_limbs = heatmap_first[:, [4,5,6,7,10,11,12,13,20,21,22,23,26,27,28,29]]
            last_real_limbs = heatmap_last[:, [4,5,6,7,10,11,12,13,20,21,22,23,26,27,28,29]]
            for t in range(T):
                t_img = interpolation(heatmap_first[:,[0,1,2,3,8,9,14,15,16,17,18,19,24,25]], heatmap_last[:,[0,1,2,3,8,9,14,15,16,17,18,19,24,25]], t)
                #print(t_img[0,0], t_img[0,1]) 50, 70
                # b, nc_center
                videos_center_x = t_img[:, 0]
                videos_center_y = t_img[:, 1]
                videos_len_x = videos_center_x - 50
                videos_len_y = videos_center_y - 70
                for i in range(nc_center //2):
                    t_img[:, i*2] = t_img[:, i*2] - videos_len_x
                    t_img[:, i*2+1] = t_img[:, i*2+1] - videos_len_y
                if t == 0:
                    for j in range(nc_limbs //2):
                        first_real_limbs[:, j*2] = first_real_limbs[:, j*2] - videos_len_x
                        first_real_limbs[:, j*2+1] = first_real_limbs[:, j*2+1] - videos_len_y
                if t == T-1:
                    for j in range(nc_limbs //2):
                        last_real_limbs[:, j*2] = last_real_limbs[:, j*2] - videos_len_x
                        last_real_limbs[:, j*2+1] = last_real_limbs[:, j*2+1] - videos_len_y
                        
                
                interpolated_video.append(t_img) # T, b, nc_center
                sequence_len_x.append(videos_len_x)
                sequence_len_y.append(videos_len_y)
            #pdb.set_trace()                
            interpolated_video = np.array(interpolated_video).transpose(1,0,2)
            # batch_size T nc_center
            ''' prepare heat map '''
            croped_video = torch.from_numpy(interpolated_video).float()
            first_real_limbs = torch.from_numpy(first_real_limbs).float()
            last_real_limbs = torch.from_numpy(last_real_limbs).float()
            croped_video, first_real_limbs, last_real_limbs, zoom_rate = Normalization(croped_video, first_real_limbs, last_real_limbs)
            #print(croped_video[0,:,10])
            croped_video = croped_video.cuda()
            first_real_limbs = first_real_limbs.cuda()
            last_real_limbs = last_real_limbs.cuda()
            first_real_limbs_encoded, _, _ = encoder(first_real_limbs) # b, nc_noise
            last_real_limbs_encoded, _, _ = encoder(last_real_limbs)
            ''' generate fake videos sequence '''
            noise = Variable(gaussian_noise.sample((mini_batch, T, nc_noise))).cuda()
            noise[:, 0, :] = first_real_limbs_encoded
            noise[:,-1, :] = last_real_limbs_encoded
            GV.initHidden(mini_batch)
            noise_T = GV(noise) # b, T*nc_noise
            noise_T = noise_T.view(mini_batch, T, nc_noise)
            ''' generate fake heat map videos '''
            fake_videos = []
            for t in range(T):
                fake_image = GI(croped_video[:, t, :], noise_T[:, t, :])
                fake_videos.append(fake_image)
                                
            fake_videos = torch.stack(fake_videos, dim=0).permute(1,0,2)  # b, T, nc_heatmap
            for t in range(T):
                fake_videos_center_x = fake_videos[:,t,0]
                fake_videos_center_y = fake_videos[:,t,1]
                fake_videos_len_x = fake_videos_center_x - 50
                fake_videos_len_y = fake_videos_center_y - 70
                for i in range(nc_heatmap // 2):
                    fake_videos[:,t,i*2] = fake_videos[:,t,i*2] - fake_videos_len_x
                    fake_videos[:,t,i*2+1] = fake_videos[:,t,i*2+1] - fake_videos_len_y
                    
            human_center = [0,1,2,3,8,9,14,15,16,17,18,19,24,25]
            f = 0
            for j in human_center:
                fake_videos[:,:,j] = croped_video[:,:,f]
                f+=1
                
            fake_video_heatmap = torch.zeros([1, T, nc_heatmap//2, 256, 256]).cuda()
            fake_video = torch.zeros([T, 256, 256, 1])
            fake_videos = fake_videos.cpu()
            #print(fake_videos[0,:,18])
            final_videos = torch.zeros([1, T, 3, 256, 256])
            #fake_video_center = torch.zeros([T, 300, 300, 1]).cuda()
            human_ctol = [2,5,10,13]
            #human_limbs = [3,6,11,14]
            for t in range(T):
                for i in range(nc_heatmap//2):
                            
                    #print(fake_videos[0,t,i*2])
#                    print(zoom_rate[0,:])
                    fake_videos[:,t,i*2] = (fake_videos[:,t,i*2] - 50.)/zoom_rate[:,t] + 50.
                    fake_videos[:,t,i*2+1] =(fake_videos[:,t,i*2+1] - 70.)/zoom_rate[:,t] + 70.
                    
                    fake_videos[:, t, i*2] = fake_videos[:, t, i*2] + torch.from_numpy(sequence_len_x[t]).float()
                    fake_videos[:, t, i*2+1] = fake_videos[:, t, i*2+1] + torch.from_numpy(sequence_len_y[t]).float()
                
                for j in human_ctol:
                    if t >=1:
                        for b in range(mini_batch):
                            if torch.abs(fake_videos[b,t,j*2] - fake_videos[b,t-1,j*2]) <=5:
                                fake_videos[b,t,j*2] = fake_videos[b,t-1,j*2]
                            if torch.abs(fake_videos[b,t,j*2+1] - fake_videos[b,t-1,j*2+1]) <=5:
                                fake_videos[b,t,j*2+1] = fake_videos[b,t-1,j*2+1]
                    
                for u in range(nc_heatmap//2):
                #print(fake_videos[0,t,0], fake_videos[0,t,1])
                    fake_video_heatmap[0,t, u, min(torch.round(fake_videos[0, t, u*2+1]).int().tolist(),255), min(torch.round(fake_videos[0, t, u*2]).int().tolist(),255)] = 1
                    fake_video_heatmap[0,t, u, min(torch.round(fake_videos[0, t, u*2+1]).int().tolist()+1,255), min(torch.round(fake_videos[0, t, u*2]).int().tolist(),255)] = 1
                    fake_video_heatmap[0,t, u, min(torch.round(fake_videos[0, t, u*2+1]).int().tolist()-1,255), min(torch.round(fake_videos[0, t, u*2]).int().tolist(),255)] = 1
                    fake_video_heatmap[0,t, u, min(torch.round(fake_videos[0, t, u*2+1]).int().tolist(),255), min(torch.round(fake_videos[0, t, u*2]).int().tolist()+1,255)] = 1
                    fake_video_heatmap[0,t, u, min(torch.round(fake_videos[0, t, u*2+1]).int().tolist(),255), min(torch.round(fake_videos[0, t, u*2]).int().tolist()-1,255)] = 1
                      
                    fake_video[t, min(torch.round(fake_videos[0, t, u*2+1]).int().tolist(),255), min(torch.round(fake_videos[0, t, u*2]).int().tolist(),255), 0] = 255
                    fake_video[t, min(torch.round(fake_videos[0, t, u*2+1]).int().tolist()+1,255), min(torch.round(fake_videos[0, t, u*2]).int().tolist(),255), 0] = 255
                    fake_video[t, min(torch.round(fake_videos[0, t, u*2+1]).int().tolist()-1,255), min(torch.round(fake_videos[0, t, u*2]).int().tolist(),255), 0] = 255
                    fake_video[t, min(torch.round(fake_videos[0, t, u*2+1]).int().tolist(),255), min(torch.round(fake_videos[0, t, u*2]).int().tolist()+1,255), 0] = 255
                    fake_video[t, min(torch.round(fake_videos[0, t, u*2+1]).int().tolist(),255), min(torch.round(fake_videos[0, t, u*2]).int().tolist()-1,255), 0] = 255
                
                
                input1 = torch.cat([videos_forground, fake_video_heatmap[:,t,:,:,:]], 1)
                result_coarse = heat_to_ori(input1)
                input2 = torch.cat([videos_background_f, videos_background_l, fake_video_heatmap[:,t,:,:,:]], 1)
                background = unet_background(input2)
                input3 = torch.cat([background, result_coarse], 1)
                final_videos[:, t, :, :, :] = unet_refine(input3)
                    
                result_coarse = result_coarse.detach()
                background = background.detach()
                final_videos = final_videos.detach()
                    
            if ind_t == 0:
                fake_video_all = fake_video.numpy()
                real_video_all = RGB_video.numpy()
                fake_video_heatmap_all = final_videos.numpy()
                RGB_image.append(RGB_first[0].cpu().numpy())
                RGB_image.append(RGB_last[0].cpu().numpy())
                fake_videos_position.append(torch.round(fake_videos[0,:,:]).cpu().detach().int().tolist())
            else:
                fake_video_all = np.concatenate((fake_video_all, fake_video.numpy()), 0) # T nc_heatmap//2, 256, 256
                real_video_all = np.concatenate((real_video_all, RGB_video.numpy()), 1) # b, T, 3, 256, 256
                fake_video_heatmap_all = np.concatenate((fake_video_heatmap_all, final_videos.numpy()), 1)
                RGB_image.append(RGB_last[0].cpu().numpy())
                fake_videos_position.append(torch.round(fake_videos[0,:,:]).cpu().detach().int().tolist())
            
        print('[%d/%d] (%s) '
          % (epoch, n_iter, timeSince(start_time))) 
        
        real_video_all = real_video_all[0].transpose(0,2,3,1) # T, 256, 256, 3
        fake_video_heatmap_all = fake_video_heatmap_all[0].transpose(0,2,3,1)
        RGB_image = np.array(RGB_image).transpose(0,2,3,1) # ing_t+1, 256, 256, 3
            
                
        save_video(fake_video_all, real_video_all, fake_video_heatmap_all, step)
        for k in range(6):
            save_image(RGB_image[k], step, k)
            
        connect = []
        for y in range(5):
            for t in range(T):
                limbSeq = [[0,1], [1,2], [2,3], [0,4], [4,5], [5,6], [0,7], [7,8], [8,9], [9,10], \
                           [10,11], [8,12], [12,13], [13,14]]
                indices = []
                for i in limbSeq: #4
                    p0 = i[0]
                    p1 = i[1]
                    r0 = fake_videos_position[y][t][p0*2+1]
                    c0 = fake_videos_position[y][t][p0*2]
                    r1 = fake_videos_position[y][t][p1*2+1]
                    c1 = fake_videos_position[y][t][p1*2]
        
                    ind = _getSparseKeypoint(r0, c0, 0, height, width, radius, var, mode)    
                    indices.extend(ind)
        
                    ind = _getSparseKeypoint(r1, c1, 0, height, width, radius, var, mode)
                    indices.extend(ind)
            
                    distance = np.sqrt((r0-r1)**2 + (c0-c1)**2)
                    sampleN = int(distance/radius)
                    if sampleN > 1:
                        for i in range(1,sampleN):
                            r = r0 + (r1-r0)*i/sampleN
                            c = c0 + (c1-c0)*i/sampleN
                            ind = _getSparseKeypoint(r, c, 0, height, width, radius, var, mode)
                            indices.extend(ind)
                
                shape = [height, width, 1]
    
                dense = np.squeeze(_sparse2dense(indices, shape)) # dense shape = (256, 256)
                connect.append(dense)
        connect = np.array(connect)
        connect = connect*255.0
        connect = connect.astype(np.uint8)
        file_path = os.path.join(dir_path, 'connectVideo_epoch-%d.mp4' % step)
        skvideo.io.vwrite(file_path, connect)
    

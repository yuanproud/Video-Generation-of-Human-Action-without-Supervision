import os
import argparse
import time
import math
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable, grad
import torch.utils.data as Data
import imageio
import scipy
from spacepy import pycdf
from skimage.morphology import square, dilation, erosion

from pose_guided_models import UAE_noFC_AfterNoise, Generator, Discriminator, GeneratorLoss

parser = argparse.ArgumentParser(description='Start trainning MoCoGAN.....')
parser.add_argument('--cuda', type=int, default=1,
                     help='set -1 when you use cpu')
parser.add_argument('--multiple-gpu', type=int, default=-1,
                     help='set 1 when you use multiple gpu')
parser.add_argument('--ngpu', type=int, default=0,
                     help='set the number of gpu you use')
parser.add_argument('--batch-size', type=int, default=8,
                     help='set batch_size, default: 16')
parser.add_argument('--niter', type=int, default=3000,
                     help='set num of iterations, default: 120000')
parser.add_argument('--pre-train', type=int, default=-1,
                     help='set 1 when you use pre-trained models')
parser.add_argument('--image-size', type=int, default=256,
                     help='Image size')

args       = parser.parse_args()
cuda       = args.cuda
multiple_gpu = args.multiple_gpu
ngpu       = args.ngpu
batch_size = args.batch_size
n_iter     = args.niter
pre_train  = args.pre_train
img_size = args.image_size

T = 32
nc = 3
nc_heatmap = 15
LAMBDA = 10
lambda_gp = 10

height = img_size
width = img_size
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
resized_path = os.path.join(current_path, 'train_data/Pose_D2_256')
videos_cdf = []
for root, dir_names, fnames in os.walk(resized_path):
    for fname in sorted(fnames):
        path = os.path.join(root, fname)
        videos_cdf.append(path)
resized_path = os.path.join(current_path, 'train_data/Pose_D2_256_Sitting_Down')
for root, dir_names, fnames in os.walk(resized_path):
    for fname in sorted(fnames):
        path = os.path.join(root, fname)
        videos_cdf.append(path)
resized_path = os.path.join(current_path, 'train_data/Pose_D2_256_Sitting')
for root, dir_names, fnames in os.walk(resized_path):
    for fname in sorted(fnames):
        path = os.path.join(root, fname)
        videos_cdf.append(path)


videos = []
videos_path = os.path.join(current_path, 'train_data/Videos_Walking_256')
for root, dir_names, fnames in os.walk(videos_path):
    for fname in sorted(fnames):
        if is_image_file(fname):
            path = os.path.join(root, fname)
            videos.append(path)
videos_path = os.path.join(current_path, 'train_data/Videos_Sitting_Down_256')
for root, dir_names, fnames in os.walk(videos_path):
    for fname in sorted(fnames):
        if is_image_file(fname):
            path = os.path.join(root, fname)
            videos.append(path)
videos_path = os.path.join(current_path, 'train_data/Videos_Sitting_256')
for root, dir_names, fnames in os.walk(videos_path):
    for fname in sorted(fnames):
        if is_image_file(fname):
            path = os.path.join(root, fname)
            videos.append(path)
            
videos_pl = []
videos_path = os.path.join(current_path, 'train_data/Segments_mat_gt_pl_Walking_256')
for root, dir_names, fnames in os.walk(videos_path):
    for fname in sorted(fnames):
        if is_image_file(fname):
            path = os.path.join(root, fname)
            videos_pl.append(path)
videos_path = os.path.join(current_path, 'train_data/Segments_mat_gt_pl_Sitting_Down_256')
for root, dir_names, fnames in os.walk(videos_path):
    for fname in sorted(fnames):
        if is_image_file(fname):
            path = os.path.join(root, fname)
            videos_pl.append(path)
videos_path = os.path.join(current_path, 'train_data/Segments_mat_gt_pl_Sitting_256')
for root, dir_names, fnames in os.walk(videos_path):
    for fname in sorted(fnames):
        if is_image_file(fname):
            path = os.path.join(root, fname)
            videos_pl.append(path)
            
print(len(videos_cdf))
print(len(videos))
print('finish load data')

''' prepare video sampling '''


# for true video
def trim(video1, video2):
    start = np.random.randint(0, video1.shape[1] - (T+1))
    end = start + T
    return video1[:, start:end, :, :, :], video2[:, start:end, :, :, :]

''' set models '''
mse = nn.L1Loss()
SL1 = nn.SmoothL1Loss()
criterion = nn.BCELoss()

if multiple_gpu == True:    
    unet_coarse = nn.DataParallel(UAE_noFC_AfterNoise(nc_heatmap+nc,3))
    unet_background = nn.DataParallel(UAE_noFC_AfterNoise(nc*2+nc_heatmap, 3))
    unet_refine = nn.DataParallel(Generator())
    dis = nn.DataParallel(Discriminator())
else:
    unet_coarse = UAE_noFC_AfterNoise(nc_heatmap+nc,3)
    unet_background = UAE_noFC_AfterNoise(nc*2+nc_heatmap, 3)
    unet_refine = Generator()
    dis = Discriminator()

''' prepare for train '''
def timeSince(since):
    now = time.time()
    s = now - since
    d = math.floor(s / ((60**2)*24))
    h = math.floor(s / (60**2)) - d*24
    m = math.floor(s / 60) - h*60 - d*24*60
    s = s - m*60 - h*(60**2) - d*24*(60**2)
    return '%dd %dh %dm %ds' % (d, h, m, s)

trained_path = os.path.join(current_path, 'trained_models_heatmap_to_original_SRGAN')
if not os.path.exists(trained_path):
    os.makedirs(trained_path)
    
dir_path = os.path.join(current_path, 'generated_videos_heatmap_to_original_SRGAN')
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    
def checkpoint(model, model_name, optimizer, epoch):
    filename = os.path.join(trained_path, model_name+'_epoch-%d' % (epoch))
    torch.save(model.state_dict(), filename + '.pkl')
    torch.save(optimizer.state_dict(), filename + '.state')

def save_image(backg, forg, fake_video_refine, real_video, real_video_center, real_back, epoch):
    
    outputdata = backg * 255.0
    outputdata = outputdata.astype(np.uint8)
    file_path = os.path.join(dir_path, 'background_epoch-%d.jpg' % epoch)
    scipy.misc.imsave(file_path, outputdata)
    
    outputdata = forg * 255.0
    outputdata = outputdata.astype(np.uint8)
    file_path = os.path.join(dir_path, 'foreground_epoch-%d.jpg' % epoch)
    scipy.misc.imsave(file_path, outputdata)
    
    outputdata = fake_video_refine * 255.0
    outputdata = outputdata.astype(np.uint8)
    file_path = os.path.join(dir_path, 'refine_fake_epoch-%d.jpg' % epoch)
    scipy.misc.imsave(file_path, outputdata)
    
    realdata = real_video * 255.0
    realdata = realdata.astype(np.uint8)
    file_path = os.path.join(dir_path, 'real_epoch-%d.jpg' % epoch)
    scipy.misc.imsave(file_path, realdata)

    realdata = real_video_center * 255.0
    realdata = realdata.astype(np.uint8)
    file_path = os.path.join(dir_path, 'real_center_epoch-%d.jpg' % epoch)
    scipy.misc.imsave(file_path, realdata)
    
    realdata = real_back * 255.0
    realdata = realdata.astype(np.uint8)
    file_path = os.path.join(dir_path, 'real_background_epoch-%d.jpg' % epoch)
    scipy.misc.imsave(file_path, realdata)

if pre_train == True:
    print('Load pre-train model')
    unet_coarse.load_state_dict(torch.load(trained_path + '/unet_coarse_epoch-10.pkl')) 
    unet_background.load_state_dict(torch.load(trained_path + '/unet_backgrounf-10.pkl'))
    unet_refine.load_state_dict(torch.load(trained_path + '/unet_refine-10.pkl'))
    dis.load_state_dict(torch.load(trained_path + '/dis-10.pkl'))   

''' adjust to cuda '''

if cuda == True:
    unet_coarse.cuda()
    unet_background.cuda()
    unet_refine.cuda()
    dis.cuda()
    mse.cuda()
    criterion.cuda()

generator_criterion = GeneratorLoss()

# setup optimizer
lr = 0.00005
lr_gan = 0.00005
betas=(0.5, 0.999)
optim_unet_coarse = optim.Adam(unet_coarse.parameters(), lr=lr, betas=betas)
optim_unet_background = optim.Adam(unet_background.parameters(), lr=lr, betas=betas)
optim_unet_refine = optim.Adam(unet_refine.parameters(), lr=lr_gan, betas=betas)
optim_dis = optim.Adam(dis.parameters(), lr=lr_gan, betas=betas)
    
''' calc grad of models '''
weight_mse = 1
torch_dataset = FaceIdPoseDataset(videos, videos_cdf, videos_pl)
dataloader = Data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True) 

''' train models '''
start_time = time.time()

for epoch in range(1, n_iter+1):
    for step, batch_data in enumerate(dataloader):
        ''' prepare real images '''
        # videos.size() => (batch_size, T, img_size, img_size, nc)
        videos_path = batch_data[0]
        videos_pl_path = batch_data[2]
        mini_batch = len(videos_path)
        start = np.random.randint(0, 160 - (T+1))
        end = start+T
        videos_t = []
        videos_pl = []
        for k in range(mini_batch):
            vid = imageio.get_reader(videos_path[k], 'ffmpeg')
            vid_pl = imageio.get_reader(videos_pl_path[k], 'ffmpeg')
            #leng = len(vid)
            al_r = []
            al_r_pl = []
            for ti in range(start, end):
                image = vid.get_data(ti)
                al_r.append(image)
                image_pl = vid_pl.get_data(ti)
                al_r_pl.append(image_pl)
                
            videos_t.append(al_r)
            videos_pl.append(al_r_pl)
        videos_t = torch.FloatTensor(videos_t)
        videos_t = videos_t / 255.0
        
        videos_pl = torch.FloatTensor(videos_pl)
        videos_pl = videos_pl / 255.0
        
        videos_cdf_path = batch_data[1]
        videos_cdf_all = np.zeros([mini_batch, T, 256, 256, nc_heatmap])
        videos_cdf_center = np.zeros([mini_batch, T, 2])
        for k in range(mini_batch):
            s = 0
            cdf = pycdf.CDF(videos_cdf_path[k]) # 1, 160, 30for t in range(160):    
            for t in range(start, end):
                for i in range(cdf['Pose'].shape[2]//2): #4
                    videos_cdf_all[k,s, int(cdf['Pose'][0, t, i*2+1]), int(cdf['Pose'][0, t, i*2]), i] = 1
                    videos_cdf_all[k,s, int(cdf['Pose'][0, t, i*2+1])+1, int(cdf['Pose'][0, t, i*2]), i] = 1
                    videos_cdf_all[k,s, int(cdf['Pose'][0, t, i*2+1])-1, int(cdf['Pose'][0, t, i*2]), i] = 1
                    videos_cdf_all[k,s, int(cdf['Pose'][0, t, i*2+1]), int(cdf['Pose'][0, t, i*2])+1, i] = 1
                    videos_cdf_all[k,s, int(cdf['Pose'][0, t, i*2+1]), int(cdf['Pose'][0, t, i*2])-1, i] = 1
                videos_cdf_center[k,s,0] = int(cdf['Pose'][0, t, 0])
                videos_cdf_center[k,s,1] = int(cdf['Pose'][0, t, 1])
                s+=1
        
        videos_cdf_all = torch.from_numpy(videos_cdf_all).permute(0, 1, 4, 2, 3).float()
        videos_cdf_center = torch.from_numpy(videos_cdf_center).int()
        video_first = videos_t[:,0,:,:,:].permute(0,3,1,2)
        video_last = videos_t[:,-1,:,:,:].permute(0,3,1,2)
        video_pl_first = videos_pl[:,0,:,:,:].permute(0,3,1,2)
        video_pl_last = videos_pl[:,-1,:,:,:].permute(0,3,1,2)
        
        videos_forground_f = video_first * video_pl_first
        videos_background_f = video_first * torch.abs(1.0 - video_pl_first)
        videos_background_l = video_last * torch.abs(1.0 - video_pl_last)
            
        one = Variable(torch.FloatTensor(videos_t.size(0), 1).fill_(0.9), requires_grad=False).cuda()
        mone = Variable(torch.FloatTensor(videos_t.size(0), 1).fill_(0.1), requires_grad=False).cuda()
        
        for _ in range(5):
            r = np.random.randint(10, T-10)
            videos_cdf = videos_cdf_all[:, r, :, :, :] # b, 15, 256, 256
            videos_t_r = videos_t.permute(0,1,4,2,3)[:,r,:,:,:]
            videos_pl_r = videos_pl.permute(0,1,4,2,3)[:,r,:,:,:]
            videos_cdf_center_r = videos_cdf_center[:,r,:]
            videos_cdf_center_f = videos_cdf_center[:,0,:]
            
            videos_forground_r = videos_t_r * videos_pl_r
            videos_background_r = videos_t_r * torch.abs(1.0 - videos_pl_r)
            if cuda == True:
                videos_cdf = videos_cdf.cuda()
                video_first = video_first.cuda()
                videos_t_r = videos_t_r.cuda()
                videos_forground_f = videos_forground_f.cuda()
                videos_forground_r = videos_forground_r.cuda()
                videos_background_f = videos_background_f.cuda()
                videos_background_l = videos_background_l.cuda()
                videos_background_r = videos_background_r.cuda()
                videos_cdf_center_r = videos_cdf_center_r.cuda()
                videos_cdf_center_f = videos_cdf_center_f.cuda()
                
            inputs = torch.cat([videos_forground_f, videos_cdf], 1)
            result_coarse = unet_coarse(inputs) # b, 3, 256, 256
            
            result_coarse_center = Variable(torch.zeros([mini_batch, 3, 60, 60])).cuda()
            videos_t_center = Variable(torch.zeros([mini_batch, 3 ,60, 60])).cuda()
            for b in range(mini_batch):
                result_coarse_center[b,:,:,:] = result_coarse[b,:,videos_cdf_center_r[b,0]-30:videos_cdf_center_r[b,0]+30, videos_cdf_center_r[b,1]-30:videos_cdf_center_r[b,1]+30]
                videos_t_center[b,:,:,:] = videos_t_r[b,:,videos_cdf_center_r[b,0]-30:videos_cdf_center_r[b,0]+30, videos_cdf_center_r[b,1]-30:videos_cdf_center_r[b,1]+30]
           
            
            unet_coarse.zero_grad()
            L1_loss = weight_mse * mse(result_coarse_center, videos_t_center) + 0.1 *mse(result_coarse, videos_forground_r)
            L1_loss.backward()
            optim_unet_coarse.step()
            
            input2 = torch.cat([videos_background_f, videos_background_l, videos_cdf], 1)
            background = unet_background(input2) 
            #print(torch.max(background))
            
            unet_background.zero_grad()
            L_loss = mse(background, videos_background_r)# + 0.1 * mse(fake_videos, videos_t_r)
            L_loss.backward()
            optim_unet_background.step()
            
            ''' train D'''
            input3 = torch.cat([background.detach(), result_coarse.detach()], 1)
            fake_videos = unet_refine(input3)
            
            dis.zero_grad()
            real_out = dis(videos_t_r).mean()
            fake_out = dis(fake_videos).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optim_dis.step()
            
            unet_refine.zero_grad()
            g_loss = generator_criterion(fake_out, fake_videos, videos_t_r)
            g_loss.backward()
            optim_unet_refine.step()
                
    if epoch % 1 == 0:
        print('[%d/%d] (%s) L1_loss_coarse %.4f, L2_Loss %.4f, L3_Loss %.4f, D_loss %.4f' #, DF_loss %.4f, G_loss %.4f'
              % (epoch, n_iter, timeSince(start_time), L1_loss.data, L_loss.data, g_loss.data, d_loss.data)) #, fake_loss.data, fake_loss_gi.data))
        save_image(background[0].cpu().data.numpy().transpose(1,2,0), result_coarse[0].cpu().data.numpy().transpose(1,2,0), fake_videos[0].cpu().data.numpy().transpose(1,2,0), videos_t_r[0].cpu().numpy().transpose(1,2,0), videos_forground_r[0].cpu().numpy().transpose(1,2,0), videos_background_r[0].cpu().numpy().transpose(1,2,0), epoch)
    
    if epoch % 20 == 0:
        lr *= 0.5
        optim_unet_coarse = optim.Adam(unet_coarse.parameters(), lr=lr, betas=betas)
        optim_unet_background = optim.Adam(unet_background.parameters(), lr=lr, betas=betas)
        optim_unet_refine = optim.Adam(unet_refine.parameters(), lr=lr, betas=betas)
        #optim_dis = optim.Adam(dis.parameters(), lr=lr, betas=betas)
        
    if epoch % 10 == 0:
        checkpoint(unet_coarse, 'unet_coarse', optim_unet_coarse, epoch)
        checkpoint(unet_background, 'unet_background', optim_unet_background, epoch)
        checkpoint(unet_refine, 'unet_refine', optim_unet_refine, epoch)
        checkpoint(dis, 'dis', optim_dis, epoch)

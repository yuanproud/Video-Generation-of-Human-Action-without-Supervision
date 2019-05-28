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
from spacepy import pycdf

from models import Discriminator_1d, Generator_1d, Encoder_1d
#from Pose_Estimation.demo.picture_demo import get_heatmap

parser = argparse.ArgumentParser(description='Start trainning MoCoGAN.....')
parser.add_argument('--cuda', type=int, default=1,
                     help='set -1 when you use cpu')
parser.add_argument('--multiple-gpu', type=int, default=-1,
                     help='set 1 when you use multiple gpu')
parser.add_argument('--ngpu', type=int, default=0,
                     help='set the number of gpu you use')
parser.add_argument('--batch-size', type=int, default=32,
                     help='set batch_size, default: 16')
parser.add_argument('--niter', type=int, default=5000,
                     help='set num of iterations, default: 120000')
parser.add_argument('--pre-train', type=int, default=-1,
                     help='set 1 when you use pre-trained models')
parser.add_argument('--image-size', type=int, default=256,
                     help='Image size')
parser.add_argument('--lambda_gp', type=float, default=10,
                     help='weight for gradient penalty')
parser.add_argument('--action', type=str, default='Walking',
                     help='Action of you want to generate')

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

T = 16
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
IMG_EXTENSIONS = [
    '.mp4', '.avi'
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

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

# for true video
def trim(video):
    start = np.random.randint(0, video.shape[1]//2)# - (T+1))
    end = start + T
    return video[:, start:end, :]

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
            traned[:,i,j*2] = torch.round((first_img[:,i,j*2] - 50.) * rate[:,i] + 50.)
            
            traned[:,i,j*2+1] = torch.round((first_img[:,i,j*2+1] - 70.) * rate[:,i] + 70.)
            
    return traned, rate
    
''' set models '''

criterion = nn.BCEWithLogitsLoss()
mse = nn.MSELoss()
kld = nn.KLDivLoss()

if multiple_gpu == True:    
    resnet = nn.DataParallel(Generator_1d(nc_center, nc_heatmap))
    encoder = nn.DataParallel(Encoder_1d(nc_limbs, nc_noise))
    dis_i = nn.DataParallel(Discriminator_1d(nc_heatmap, 1))
else:
    resnet = Generator_1d(nc_center, nc_heatmap)
    encoder = Encoder_1d(nc_limbs, nc_noise)
    dis_i = Discriminator_1d(nc_heatmap, 1)
 
''' prepare for train '''
def timeSince(since):
    now = time.time()
    s = now - since
    d = math.floor(s / ((60**2)*24))
    h = math.floor(s / (60**2)) - d*24
    m = math.floor(s / 60) - h*60 - d*24*60
    s = s - m*60 - h*(60**2) - d*24*(60**2)
    return '%dd %dh %dm %ds' % (d, h, m, s)

trained_path = os.path.join(current_path, 'trained_models_%s'%action)
if not os.path.exists(trained_path):
    os.makedirs(trained_path)
    
dir_path = os.path.join(current_path, 'generated_videos_%s'%action)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    
def checkpoint(model, model_name, optimizer, epoch):
    filename = os.path.join(trained_path, model_name+'_epoch-%d' % (epoch))
    torch.save(model.state_dict(), filename + '.pkl')
    torch.save(optimizer.state_dict(), filename + '.state')

def save_video(fake_video, encoded_video, real_video, epoch):
    
    outputdata = fake_video.astype(np.uint8)
    file_path = os.path.join(dir_path, 'fakeVideo_epoch-%d.mp4' % epoch)
    skvideo.io.vwrite(file_path, outputdata)
    
    outputdata = encoded_video.astype(np.uint8)
    file_path = os.path.join(dir_path, 'encodedVideo_epoch-%d.mp4' % epoch)
    skvideo.io.vwrite(file_path, outputdata)
    
    realdata = real_video.astype(np.uint8)
    file_path = os.path.join(dir_path, 'realVideo_epoch-%d.mp4' % epoch)
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
    print('Load pre-train model')
    resnet.load_state_dict(torch.load(trained_path + '/unet_epoch-5000.pkl'))    
    dis_i.load_state_dict(torch.load(trained_path + '/dis_i_epoch-5000.pkl'))  
    encoder.load_state_dict(torch.load(trained_path + '/encoder_epoch-5000.pkl'))

''' adjust to cuda '''

if cuda == True:
    resnet.cuda()
    dis_i.cuda()
    encoder.cuda()
    criterion.cuda()


# setup optimizer
lr = 0.0001
lr_dis = 0.0001
betas=(0.5, 0.999)

optim_g_v = optim.Adam(resnet.parameters(), lr=lr, betas=betas)
optim_dis_i = optim.Adam(dis_i.parameters(), lr=lr_dis, betas=betas)
optim_encoder = optim.Adam(encoder.parameters(), lr=lr_dis, betas=betas)
''' calc grad of models '''

torch_dataset = FaceIdPoseDataset(videos)
dataloader = Data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True) 
uniform_noise = torch.distributions.normal.Normal(0,1.)
''' train models '''
start_time = time.time()

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
        
        for _ in range(5):
            heat_map = torch.from_numpy(trim(videos)) # b, T, 30
            heat_map, zoom_rate = Normalization(heat_map)
            ''' prepare heat map '''
            real_image = Variable(heat_map.view(mini_batch*T, nc_heatmap), requires_grad=False).cuda()
            # b, 30
            ''' generate fake heat map videos '''
            real_center = real_image[:, [0,1,2,3,8,9,14,15,16,17,18,19,24,25]]
            noise = Variable(uniform_noise.sample((mini_batch*T, nc_noise)), requires_grad=True).cuda()
            fake_image = resnet(real_center, noise)
            # t_video size : b, 30
            ''' encode '''
            real_limbs = real_image[:, [4,5,6,7,10,11,12,13,20,21,22,23,26,27,28,29]]
            encoded, mu, logvar = encoder(real_limbs)
            encoded_image = resnet(real_center, encoded.detach())
            ''' train Discriminator_I '''
            for p in dis_i.parameters():
                p.requires_grad = True
                
            dis_i.zero_grad()
            fake_loss_di = torch.mean(dis_i(fake_image.detach()))
            real_loss_di = -1 * torch.mean(dis_i(real_image))
            gradient_penalty_i = calc_gradient_penalty(dis_i, real_image.data, fake_image.detach().data, mini_batch)
            
            di_loss = fake_loss_di + real_loss_di + lambda_gp * gradient_penalty_i
            di_loss.backward()
            optim_dis_i.step()
            
            ''' train course UNet '''
            for p in dis_i.parameters():
                p.requires_grad = False
            
            resnet.zero_grad()
            fake_loss_gi = -1 * torch.mean(dis_i(fake_image))
            
            mse_err = mse(encoded_image, real_image)
            
            fake_loss = fake_loss_gi + mse_err
            fake_loss.backward()
            optim_g_v.step()
            
            encoder.zero_grad()
            kld_err = -0.5 * torch.sum(1+logvar - mu.pow(2) - logvar.exp())
            kld_err.backward()
            optim_encoder.step()
            
    if epoch % 5 == 0:
        print('[%d/%d] (%s) Gi_mean %.4f Di_fake_mean %.4f Di_real_mean %.4f MSE_Err %.4f KLD_Err %.4f'
              % (epoch, n_iter, timeSince(start_time), fake_loss_gi.data, fake_loss_di.data, real_loss_di.data, mse_err.data, kld_err.data))   
    if epoch % 5 == 0:
        real_image = real_image.view(-1, T, nc_heatmap).data.cpu()
        fake_image = fake_image.view(-1, T, nc_heatmap).data.cpu()
        encoded_image = encoded_image.view(-1, T, nc_heatmap).data.cpu()
        for j in range(nc_heatmap //2):
            real_image[:,:,j*2] = torch.round((real_image[:,:,j*2] - 50)/zoom_rate) + 50
            real_image[:,:,j*2+1] = torch.round((real_image[:,:,j*2+1] - 70)/zoom_rate) + 70
            fake_image[:,:,j*2] = torch.round((fake_image[:,:,j*2] - 50)/zoom_rate) + 50
            fake_image[:,:,j*2+1] = torch.round((fake_image[:,:,j*2+1] - 70)/zoom_rate) + 70
            encoded_image[:,:,j*2] = torch.round((encoded_image[:,:,j*2] - 50)/zoom_rate) + 50
            encoded_image[:,:,j*2+1] = torch.round((encoded_image[:,:,j*2+1] - 70)/zoom_rate) + 70
                   
        real_image = real_image.numpy()
        fake_image = fake_image.numpy()
        encoded_image = encoded_image.numpy()
        
        max_fake = max(int(np.max(fake_image)), int(np.max(real_image)), int(np.max(encoded_image)))
        real_all = np.zeros([T, max_fake+5, max_fake+5])
        fake_all = np.zeros([T, max_fake+5, max_fake+5])
        encoded_all = np.zeros([T, max_fake+5, max_fake+5])
        
        for t in range(T):    
            for i in range(cdf['Pose'].shape[2]//2): #4
                fake_all[t, int(fake_image[0,t, i*2+1]), int(fake_image[0,t, i*2])] = 255
                fake_all[t, int(fake_image[0,t, i*2+1])+1, int(fake_image[0,t, i*2])] = 255
                fake_all[t, int(fake_image[0,t, i*2+1])-1, int(fake_image[0,t, i*2])] = 255
                fake_all[t, int(fake_image[0,t, i*2+1]), int(fake_image[0,t, i*2])+1] = 255
                fake_all[t, int(fake_image[0,t, i*2+1]), int(fake_image[0,t, i*2])-1] = 255
                
                encoded_all[t, int(encoded_image[0,t, i*2+1]), int(encoded_image[0,t, i*2])] = 255
                encoded_all[t, int(encoded_image[0,t, i*2+1])+1, int(encoded_image[0,t, i*2])] = 255
                encoded_all[t, int(encoded_image[0,t, i*2+1])-1, int(encoded_image[0,t, i*2])] = 255
                encoded_all[t, int(encoded_image[0,t, i*2+1]), int(encoded_image[0,t, i*2])+1] = 255
                encoded_all[t, int(encoded_image[0,t, i*2+1]), int(encoded_image[0,t, i*2])-1] = 255
                
                real_all[t, int(real_image[0,t, i*2+1]), int(real_image[0,t, i*2])] = 255
                real_all[t, int(real_image[0,t, i*2+1])+1, int(real_image[0,t, i*2])] = 255
                real_all[t, int(real_image[0,t, i*2+1])-1, int(real_image[0,t, i*2])] = 255
                real_all[t, int(real_image[0,t, i*2+1]), int(real_image[0,t, i*2])+1] = 255
                real_all[t, int(real_image[0,t, i*2+1]), int(real_image[0,t, i*2])-1] = 255
            
        save_video(fake_all, encoded_all, real_all, epoch)
    
    if epoch % 500 == 0:
        for p in dis_i.parameters():
            p.requires_grad = True
            
        lr *= 0.5
        lr_dis *= 0.5
        optim_g_v = optim.Adam(resnet.parameters(), lr=lr, betas=betas)
        optim_dis_i = optim.Adam(dis_i.parameters(), lr=lr_dis, betas=betas)
        optim_encoder = optim.Adam(encoder.parameters(), lr=lr_dis, betas=betas)

    if epoch % 500 == 0:
        checkpoint(resnet, 'unet', optim_g_v, epoch)
        checkpoint(encoder, 'encoder', optim_encoder, epoch)
        checkpoint(dis_i, 'dis_i', optim_dis_i, epoch)

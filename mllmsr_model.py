import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel, SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
import torch.nn as nn
from torchvision.utils import save_image
import copy
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, tokenizer_image_token_batch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.model.builder import load_pretrained_model
from basicsr.archs.lpips import LPIPS




def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)


@MODEL_REGISTRY.register()
class MLLMSR(SRGANModel):
    """RealESRGAN Model"""
    def __init__(self, opt):
        super(MLLMSR, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()
        self.queue_size = opt['queue_size']
        self.scaler = torch.cuda.amp.GradScaler()
        self.loss_lr = nn.CrossEntropyLoss()
        self.loss_hr = nn.CrossEntropyLoss()
        self.lpips_loss = LPIPS(pretrained=False, net='vgg16', lpips=True, spatial=False, pnet_rand=False, pnet_tune=True, use_dropout=True, eval_mode=True, latent=True, in_chans=4, verbose=True)
        self.lpips_loss.load_state_dict(torch.load('/home/work/project/mllmsr/basicsr/archs/vgg16_sdturbo_lpips.pth', weights_only=True), strict=False)
        self.lpips_loss.cuda()
        for p in self.lpips_loss.parameters():
            p.requires_grad = False


    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, 'queue size should be divisible by batch size'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        if self.is_train:
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            # self.gt = torch.cat([self.gt[:1]]*4, 0)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt_usm, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale1 = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale1 = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale1 = 1
            mode1 = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale1, mode=mode1)
            # noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out, sigma1, gray_noise1 = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out, sigma1, gray_noise1 = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p1 = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p1)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale2 = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale2 = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale2 = 1
            mode2 = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale2), int(ori_w / self.opt['scale'] * scale2)), mode=mode2)
            # noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out, sigma2, gray_noise2 = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out, sigma2, gray_noise2 = random_add_poisson_noise_pt(out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p2 = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p2)
            else:
                # JPEG compression
                jpeg_p2 = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p2)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = F.interpolate(self.lq, size=(ori_h, ori_w), mode='bicubic')
            
            self.textvalid = data['textvalid']
            self.labels = data['labels']
            self.input_ids = data['input_ids']


            fixed_que = ["{DEFAULT_IMAGE_TOKEN} This is the image. Can you Describe the quality of the image? ANS: "] * self.gt.shape[0]
            fixed_que_good = ["{DEFAULT_IMAGE_TOKEN} This is the image. Can you Describe the quality of the image? ANS: "] * self.gt.shape[0]
            for i in range(self.gt.shape[0]):
                text = "The image was first resized using " + str(mode1) + " interpolation with a scale factor of " + str(scale1) + ", followed by the additional noise with sigma of " + str(sigma1[i].item()) + ' and JPEG compression wigh a quality factor of ' + str(jpeg_p1[i].item()) + '. This process was then repeated: the image was resized using ' + str(mode2) + ' interpolation with a scale factor of ' + str(scale2) + ', followed by the additional noise with a sigma '+ str(sigma2[i].item()) + ' and JPEG compression with a quality factor of ' + str(jpeg_p2[i].item())+'.' 
                text_good = "The image did not undergo degradation. It was clean, sharp, visually pleasing, high-quality, and ultra-resolution."
                fixed_que[i] += text
                fixed_que_good[i] += text_good
            try:
                input_ids, labels = tokenizer_image_token_batch(fixed_que, self.net_g.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt", return_labels=True)
            except:
                input_ids, labels = tokenizer_image_token_batch(fixed_que, self.net_g.module.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt", return_labels=True)
            try:
                input_ids_good, labels_good = tokenizer_image_token_batch(fixed_que_good, self.net_g.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt", return_labels=True)
            except:
                input_ids_good, labels_good = tokenizer_image_token_batch(fixed_que_good, self.net_g.module.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt", return_labels=True)
                
            self.qual_input_ids = input_ids
            self.qual_labels = labels
            self.qual_labels_good = labels_good
            
            
        else:
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(MLLMSR, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def optimize_parameters(self, current_iter):
        l1_gt = self.gt_usm
        B = self.lq.shape[0]

        
        ## Generator
        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        latent_sr, latent_y, timestep_pred, latent_lr, latent_noise, llava_loss =self.net_g(self.lq, l1_gt, self.input_ids, self.labels, self.textvalid, self.qual_input_ids, self.qual_labels, self.qual_labels_good)
        #
        l_t = F.l1_loss(timestep_pred, torch.abs(self.lq - l1_gt).mean((1,2,3))*3000) * 0.01
        # l_t = F.l1_loss(timestep_pred, ((torch.abs(self.lq-l1_gt).mean((1,2,3)) / torch.where(l1_gt<0.5, 1-l1_gt, l1_gt).mean((1,2,3))) * 2000.)) * 0.01
        #
        l_pix = F.l1_loss(latent_sr, latent_y.detach(), reduction='mean') * 0.25
        l_noise = F.l1_loss(latent_noise, latent_y.detach(), reduction='mean') * 0.25
        l_lpips = self.lpips_loss(latent_sr, latent_y, do_norm=True).mean() * 1.0
        l_g_dis = self.net_d(latent_lr, latent_sr)
        l_g_dis = self.cri_gan(l_g_dis, True, is_disc=False).mean()
        
        
        loss = (l_pix.mean() + l_t.mean() + l_g_dis.mean() + l_lpips.mean() + llava_loss.mean()) * 0.00001 #
        loss.backward()
        self.optimizer_g.step()
        
        
        ## Discriminator
        for p in self.net_d.parameters():
            p.requires_grad = True
        self.optimizer_d.zero_grad()
        l_d_t = self.net_d(latent_lr, latent_y)
        l_d_t = self.cri_gan(l_d_t, True, is_disc=True).mean()
        
        l_d_f = self.net_d(latent_lr, latent_sr.detach())
        l_d_f = self.cri_gan(l_d_f, False, is_disc=True).mean() 
        
        loss_d = (l_d_t + l_d_f) * 0.00001
        loss_d.backward()
        self.optimizer_d.step()        
        loss_dict = OrderedDict()
        loss_dict['l_pix'] = l_pix
        loss_dict['l_noise'] = l_noise
        loss_dict['l_lpips'] = l_lpips
        loss_dict['l_t'] = l_t
        loss_dict['l_g'] = l_g_dis
        loss_dict['l_d'] = (l_d_t + l_d_f)
        loss_dict['llm'] = (llava_loss.mean())
        # loss_dict['dmd'] = (dmd_loss.mean())
        
        self.log_dict = self.reduce_loss_dict(loss_dict)

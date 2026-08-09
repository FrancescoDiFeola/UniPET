import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks3d as networks
from monai.inferers import sliding_window_inference


def FalseAwarePETLoss(pred, target, min_weight=0.1, power=2, under_penalty=9, over_penalty=9, thresh=0.75):
    """
    Penalizes:
    - Underestimation in bright (high-SUV) regions → avoid false negatives
    - Overestimation in dark (low-SUV) regions → avoid false positives
    """
    
    # Base weighting from intensity
    weights = torch.clamp(target, 0, 1) ** power  # Raise to power (e.g., 2 → quadratic emphasis on bright voxels).
    weights = weights + min_weight  # Add min_weight so even background (low SUV) regions contribute to the loss.

    # Masks
    under_mask = (pred < target).float()  #  1 where the prediction underestimates the target (potential false negative).
    over_mask = (pred > target).float()  #  1 where the prediction overestimates the target (potential false positive).

    # Bright region: strong penalty if underpredicted (false negatives)
    bright_region = (target >= thresh).float()  # Identify bright (tumor/high-SUV) regions based on bright_thresh.
    under_weight = 1 + under_penalty * (under_mask * bright_region)  # The weight becomes 1 + penalty only where both conditions are met.

    # Dark region: strong penalty if overpredicted (false positives)
    dark_region = (target < thresh).float()  # Identify dark (background/low-SUV) regions based on dark_thresh.
    over_weight = 1 + over_penalty * (over_mask * dark_region) # The weight becomes 1 + penalty only where both conditions are met.

    # Combine masks into penalty
    ratio_weight = under_weight * over_weight

    # Final weighted loss
    diff = torch.abs(pred - target)
    loss = weights * ratio_weight * diff  # weights: intensity-based weight, ratio_weight: penalty weight based on under/overprediction

    return loss.mean()



def intensity_weighted_L1(pred, target, min_weight=0.1, power=2):
    """
    Compute an L1 loss weighted by PET intensity (SUV range already normalized to [0, 1]).

    Args:
        pred: predicted PET (normalized)
        target: ground truth PET (normalized)
        min_weight: minimum weight for low-uptake regions to avoid ignoring background completely
        power: exponent for emphasizing bright areas (e.g., 1 = linear, 2 = quadratic)
    """
    # Use normalized SUV as weight base
    weights = torch.clamp(target, 0, 1) ** power

    # Add a small constant to keep low-uptake regions in the loss
    weights = weights + min_weight

    loss = weights * torch.abs(pred - target)
    return loss.mean()


class Pix2Pix3dModel(BaseModel):
    def name(self):
        return 'Pix2Pix3dModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain # Train --> self.isTrain == True
                                   # Test --> self.isTrain == False
        # define tensors (per i dati di input)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.depthSize, opt.fineSize, opt.fineSize) # tensori allocati su GPU
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.depthSize, opt.fineSize, opt.fineSize)

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        #self.image_paths = input['A_paths' if AtoB else 'B_paths']
        
    def set_labels(self, labels): 
        self.labels = labels # memorizza le etichette per usarle durante il training 

    def forward(self):
        self.real_A = Variable(self.input_A) # CT_real
        #self.fake_B = self.netG.forward(self.real_A)
        self.fake_B, self.fake_class = self.netG.forward(self.real_A)  # Estrarre immagine e classe
        self.real_B = Variable(self.input_B)

    
    def test_sliding(self):
        with torch.no_grad():
            self.real_A = self.input_A

            fake_B = sliding_window_inference(
                self.real_A, roi_size=(32, 32, 32), sw_batch_size=3,
                predictor=self.netG, overlap=0.625, mode='gaussian',
                sigma_scale=0.125, sw_device="cuda", device="cuda"  # padding_mode='reflect'
            )  # roi:dimensione patch, sw_batch_size: numero di batch con modalità sliding window che si calcola come: batch size*num_patch
               # predictor: generatore della GAN, overlap: 62.5% di sovrapposizione, mode = 'gaussian'  è un peso gaussiano per blending (assegna pesi maggiori alcentro del patch= piu affidabile),
               # sigma scale è un parametro della gaussiana: è un coefficiente moltiplicativo che determina quanto "velocemente" il peso della gaussiana decade dal centro verso i bordi della patch.
 
            self.fake_B = fake_B.clip(0, 1)
            self.real_B = self.input_B

    # no backprop gradients
    def test(self):
        #print(f"dimensioni input_A: {self.input_A.size()}")  # dimensione tensore input
        #self.real_A = Variable(self.input_A, volatile=True)
        #print(f"dimensioni real_A: {self.real_A.size()}")
        #self.fake_B, self.fake_class = self.netG.forward(self.real_A)  # immagine ricostruita e predizione
        #print(f"dimensioni fake_B: {self.fake_B.size()}")
        #print(f"dimensioni fake_class: {self.fake_class.size()}")
        #self.real_B = Variable(self.input_B, volatile=True)
        #print(f"dimensioni real_B: {self.real_B.size()}")

        with torch.no_grad():
            self.real_A = self.input_A
            # Calcolo delle previsioni
            self.fake_B, self.fake_class = self.netG.forward(self.real_A)
            #self.fake_B = self.netG.forward(self.real_A) # per provare rete di rebecca
            self.real_B = self.input_B

    # get image paths
    def get_image_paths(self):
        return "blksdf"
        #return self.image_paths

    def backward_D(self): # il discriminatore distingue tra immagini reale e generate. 
                          # poichè non gestisce la classificazione, non ho bisogno delle etichette 
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1)) # Usare solo l'immagine generata
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False) # Usa la versione senza classificazione

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True) # usa la versione senza classificazione

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self): # il generatore deve tener conto sia della GAn loss dia della classifiction loss
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1) # Usare solo l'immagine generata
        pred_fake = self.netD.forward(fake_AB)

        self.loss_G_GAN, self.classification_loss = self.criterionGAN.compute_with_classification(pred_fake, True, self.fake_class, self.labels)

        # Second, G(A) = B (L1 Loss)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        
        
        # Somma delle perdite: GAN loss, L1 loss e classification loss
        self.loss_G = (self.loss_G_GAN + self.loss_G_L1 + self.classification_loss)
        torch.autograd.set_detect_anomaly(True)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('Classification_loss', self.classification_loss),
                            ('Total_G_Loss', self.loss_G.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item()),
                            ('Total_D_Loss', self.loss_D.item())
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im3d(self.real_A.data)
        fake_B = util.tensor2im3d(self.fake_B.data)
        real_B = util.tensor2im3d(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr



    def load_best_network(self):
        best_G = os.path.join(self.opt.checkpoints_dir, self.opt.name,'best_net_G.pth')
        best_D = os.path.join(self.opt.checkpoints_dir, self.opt.name,'best_net_D.pth')
        print('load best network')
        self.netG.load_state_dict(torch.load(best_G))
        self.netD.load_state_dict(torch.load(best_D))

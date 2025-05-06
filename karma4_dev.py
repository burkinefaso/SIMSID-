import torch
from dataloader.dataloader_karma4 import KarmaDataset
from configs.base import BaseConfig
import multiprocessing



class MemoryMatrixBlockConfig:
    memory_layer_type = 'default'  # ['default', 'dim_reduce']
    num_memory = 4
    num_slots = 500
    slot_dim = 2048  # 'dim_reduce' için kullanılır
    shrink_thres = 5
    mask_ratio = 1.0


class InpaintBlockConfig:
    use_memory_queue = True
    use_inpaint = True
    num_slots = 200
    memory_channel = 128 * 4 * 4  
    shrink_thres = 5
    drop = 0. # Transformer katmanındaki MLP'de kullanılan
    mask_ratio = 1.0


class Config(BaseConfig):
    memory_config = MemoryMatrixBlockConfig()
    inpaint_config = InpaintBlockConfig()

    def __init__(self):
        super(Config, self).__init__()

        #---------------------
        # Eğitim Parametreleri
        #---------------------
        self.print_freq = 10
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.epochs = 1000
        self.lr = 1e-4  # Öğrenme oranı
        self.batch_size = 16
        self.test_batch_size = 2
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        self.scheduler_args = dict(T_max=300, eta_min=self.lr * 0.1)
        self.val_freq = 1

        # GAN
        self.gan_lr = 1e-4
        self.discriminator_type = 'basic'
        self.enable_gan = 0  # 100
        self.lambda_gp = 10.
        self.size = 4
        self.num_layers = 4
        self.n_critic = 2
        self.sample_interval = 1000
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR
        self.scheduler_args_d = dict(T_max=200, eta_min=self.lr * 0.2)

        # Model Parametreleri
        self.img_size = 128  # 256x256 görüntüler için
        self.normalize_tanh = True  # True ise [-1, 1], False ise [0, 1] aralığında normalize edilir.

        self.num_patch = 2
        self.level = 4
        self.dist = True
        self.ops = ['concat', 'concat', 'none', 'none']
        self.decoder_memory = ['V1', 'V1', 'none', 'none']

        # Kayıp Fonksiyonu Ağırlıkları
        self.t_w = 0.01
        self.recon_w = 10.
        self.dist_w = 0.001
        self.g_w = 0.005
        self.d_w = 0.005

        self.use_memory_inpaint_block = True
        
        self.positive_ratio = 1.0
                                                                                                   
        # Veri Yolları
        self.data_root = r'C:\Users\BURKINEFASO\SIMSID_PLUS\data\karma4'
        self.train_dataset = KarmaDataset(
            root=self.data_root + '/train',
            train=True,
            img_size=(self.img_size, self.img_size),
            normalize_tanh=self.normalize_tanh
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False
        )
        
        self.val_dataset = KarmaDataset(
            root=self.data_root + '/val',
            train=False,
            img_size=(self.img_size, self.img_size),
            normalize_tanh=self.normalize_tanh,
            full=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False
        )

        self.test_dataset = KarmaDataset(
            root=self.data_root + '/test',
            train=False,
            img_size=(self.img_size, self.img_size),
            normalize_tanh=self.normalize_tanh,
            full=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False
        )
        

        
        
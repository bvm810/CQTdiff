import os
from datetime import date
import torch
from src.model import Unet_CQT
from src.utils.setup import load_ema_weights
from src.sde import  VE_Sde_Elucidating

class Exp_Base():
    def __init__(
        self, args
    ):
        self.args=args
        #choose gpu as the device if possible
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.args.architecture=="unet_CQT_nobias":
            self.model=Unet_CQT(self.args, self.device).to(self.device)
            self.model=load_ema_weights(self.model,os.path.join(args.model_dir, args.inference.checkpoint))
        else:
            raise NotImplementedError

        if args.sde_type=='VE_elucidating':
            self.diff_parameters=VE_Sde_Elucidating(self.args.diffusion_parameters, self.args.diffusion_parameters.sigma_data)
        else:
            raise NotImplementedError

        torch.backends.cudnn.benchmark = True

        today=date.today() 
        self.path_sampling=os.path.join(args.model_dir,self.args.inference.mode+today.strftime("%d_%m_%Y"))
        if not os.path.exists(self.path_sampling):
            os.makedirs(self.path_sampling)

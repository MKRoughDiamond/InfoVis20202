import torch
from torchvision.datasets import *
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import time
from load_model import *
from sklearn.manifold import TSNE

class PyTorchModule:
    def __init__(self, dataset_name='MNIST', modelname="betaB", tsne_length=100, vis_B_shape=[5,5], vis_C_length=11, delta=0.05):
        self.param = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.set_model(modelname)
        self.param['model_name'] = modelname

        if dataset_name == 'MNIST':
            self.dataset = get_dataset(self.param['model_name'])
            self.loader = DataLoader(self.dataset,batch_size=100,shuffle=True)
        else:
            raise NotImplementedError

        self.small_data = self._data_load(tsne_length)
        x, y = self.small_data
        recon_img, _, latent = self.model(x)
        self.mins = torch.min(latent,0)[0]
        self.maxs = torch.max(latent,0)[0]

        self.param['tsne_length'] = tsne_length
        self.param['vis_B_shape'] = vis_B_shape
        self.param['vis_C_length'] = vis_C_length
        self._set_tile()
        self.param['delta'] = delta
        self.tsne_model = TSNE
        self.param['target_idx'] = None

    def get_params(self):
        res_contents = {
            'model_name': self.param['model_name'],
            'tsne_length': self.param['tsne_length'],
            'vis_B_shape': self.param['vis_B_shape'],
            'vis_C_length': self.param['vis_C_length'],
            'delta': self.param['delta']
        }
        return res_contents

    
    def set_param(self, param_name, value):
        if param_name is None or param_name not in list(self.param.keys()) or value is None:
            return False
        try:
            if '.' in value:
                self.param[param_name] = float(value)
            else:
                self.param[param_name] = int(value)
        except Exception:
            self.param[param_name] = value

        if param_name == 'tsne_length':
            self.small_data = self._data_load(self.param[param_name])
        elif param_name == 'vis_B_shape':
            try:
                self.param[param_name] = list(map(lambda x: int(x), self.param[param_name]))
            except Exception:
                pass
            self._set_tile()
        elif param_name == 'vis_C_length':
            self._set_tile()
        elif param_name == 'model_name':
            self.set_model(self.param[param_name])
        elif param_name == 'target_idx':
            if len(value)!=2:
                return False
        return True


    def _set_tile(self):
        self.tile_x = torch.arange(self.param['vis_B_shape'][1]).float().unsqueeze(0).repeat(self.param['vis_B_shape'][0],1).to(self.device)-self.param['vis_B_shape'][1]//2
        self.tile_y = torch.arange(self.param['vis_B_shape'][1]).float().unsqueeze(0).repeat(self.param['vis_B_shape'][1],1).transpose(0,1).to(self.device)-self.param['vis_B_shape'][1]//2
        self.linear_c = torch.arange(self.param['vis_C_length']).float().unsqueeze(0).to(self.device)-self.param['vis_C_length']//2


    def set_model(self,modelname):
        if modelname is None:
            self.model = get_model().to(self.device).eval()
        else:
            self.model = get_model(modelname).to(self.device).eval()


    def _data_load(self,length):
        data_x = [torch.tensor([],device=self.device) for _ in range(10)]
        data_y = torch.tensor([i//(length//10) for i in range(length)],device=self.device,dtype=torch.int32)
        cnt_y = torch.zeros(10,device=self.device)
        for x,y in self.loader:
            x = x.to(self.device)
            y = y.to(self.device)
            for i,label in enumerate(y):
                data_x[label]=torch.cat((data_x[label],x[i].unsqueeze(0)),0)
                cnt_y[label]+=1
            if len(cnt_y[cnt_y<length//10]) == 0:
                data_x = list(map(lambda x: x[:length//10],data_x))
                data_x = torch.cat(data_x,dim=0)
                break
        return data_x, data_y


    def _tile_gen(self,latent):
        latent = torch.tensor(latent,device=self.device,dtype=torch.float32)
        dup = latent.unsqueeze(-1).unsqueeze(-1).repeat(1,self.param['vis_B_shape'][0],self.param['vis_B_shape'][1])
        dup[self.param['target_idx'][0]]+=self.tile_x*self.param['delta']*(self.maxs[self.param['target_idx'][0]]-self.mins[self.param['target_idx'][0]])
        dup[self.param['target_idx'][1]]+=self.tile_y*self.param['delta']*(self.maxs[self.param['target_idx'][1]]-self.mins[self.param['target_idx'][1]])
        dup = dup.view(-1,self.param['vis_B_shape'][0]*self.param['vis_B_shape'][1]).transpose(0,1)
        return dup


    def tile_imgs_gen(self,latent):
        if self.mins is None or self.maxs is None:
            return None, None
        dup = self._tile_gen(latent)
        recon_img = self.model.decoder(dup)
        return recon_img.squeeze().cpu().detach().numpy(), dup.cpu().detach().numpy()


    def _linear_gen(self,latent):
        latent = torch.tensor(latent,device=self.device,dtype=torch.float32)
        dup = latent.unsqueeze(-1).unsqueeze(-1).repeat(1,len(latent),self.param['vis_C_length'])
        for i in range(len(latent)):
            dup[i,i]+=(self.linear_c*self.param['delta']*(self.maxs[i]-self.mins[i])).squeeze()
        dup = dup.view(-1,self.param['vis_C_length']*len(latent)).transpose(0,1)
        return dup


    def linear_imgs_gen(self, latent):
        if self.mins is None or self.maxs is None:
            return None, None
        dup = self._linear_gen(latent)
        recon_img = self.model.decoder(dup)
        return recon_img.squeeze().cpu().detach().numpy(), dup.cpu().detach().numpy()
        


    def get_min_max(self):
        if self.mins is None or self.maxs is None:
            return None,None
        return self.mins.cpu().detach().numpy().tolist(), self.maxs.cpu().detach().numpy().tolist()


    def tsne_visualization(self):
        x, y = self.small_data
        recon_img, _, latent = self.model(x)
        self.mins = torch.min(latent,0)[0]
        self.maxs = torch.max(latent,0)[0]
        recon_img = recon_img.squeeze().cpu().detach().numpy()
        latent = latent.cpu().detach().numpy()
        tsne = self.tsne_model(n_components=2).fit_transform(latent)
        return recon_img, latent, tsne, y.cpu().detach().numpy()


    def enc_img(self,img):
        x = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(self.device)
        recon_img, _, latent = self.model(x)
        return latent[0].cpu().detach().numpy()

import torch
from torchvision.datasets import *
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import time
from load_model import *
from sklearn.manifold import TSNE

class PyTorchModule:
    def __init__(self,dataset_name,modelname=None,preload=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.set_model(modelname)

        if dataset_name =='MNIST':
            self.dataset = get_dataset()
            self.loader = DataLoader(self.dataset,batch_size=100,shuffle=True)
        else:
            raise NotImplementedError

        if preload:
            self.small_data = self._data_load(100)
            self.tsne_length = 100
        else:
            self.small_data = None
            self.tsne_length = None

        self.mins = None
        self.maxs = None



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


    def get_min_max(self):
        if self.mins is None or self.maxs is None:
            return None,None
        return self.mins.cpu().detach().numpy().tolist(), self.maxs.cpu().detach().numpy().tolist()


    def tsne_visualization(self,length):
        if self.small_data is not None or length != self.tsne_length:
            x, y = self.small_data
        else:
            x, y = self._data_load(length)
            self.tsne_length = length
        recon_img, _, latent = self.model(x)
        self.mins = torch.min(latent,0)[0]
        self.maxs = torch.max(latent,0)[0]
        recon_img = recon_img.squeeze().cpu().detach().numpy()
        latent = latent.cpu().detach().numpy()
        tsne = TSNE(n_components=2).fit_transform(latent)
        return recon_img, latent, tsne, y.cpu().detach().numpy()

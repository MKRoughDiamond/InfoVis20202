from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import argparse
from PIL import Image
from io import BytesIO
import base64
from DL import *
import time
import numpy as np


class DLModelServer(BaseHTTPRequestHandler):
    def _img_to_base64(self,img):
        with BytesIO() as buf:
            img.save(buf, 'jpeg')
            imgbyte = buf.getvalue()
        return base64.b64encode(imgbyte).decode()


    def _set_headers_failed(self):
        self.send_response(400)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin','*')
        self.end_headers()
        self.wfile.write(bytes(json.dumps({}),'utf-8'))


    def _set_headers_not_found(self):
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin','*')
        self.end_headers()
        self.wfile.write(bytes(json.dumps({}),'utf-8'))


    def _set_headers_success(self):
        self.send_response(200,"ok")
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin','*')
        self.end_headers()


    def _gen_imgs(self,latent,res_contents):
        imgs, latents = ptmodule.tile_imgs_gen(latent)
        if imgs is None or latents is None:
            self._set_headers_failed()
            return
        for i in range(len(imgs)):
            img = Image.fromarray(imgs[i]*255).convert('RGB')
            res_contents['tile'].append({
                'latent': latents[i].tolist(),
                'img': self._img_to_base64(img)
            })
        res_contents['target']=res_contents['tile'][len(imgs)//2]
        imgs, latents = ptmodule.linear_imgs_gen(latent)
        if imgs is None or latents is None:
            self._set_headers_failed()
            return
        for i in range(len(imgs)):
            img = Image.fromarray(imgs[i]*255).convert('RGB')
            res_contents['linear'].append({
                'latent': latents[i].tolist(),
                'img': self._img_to_base64(img)
            })
        return res_contents


    def do_GET(self):
        try:
            global ptmodule
            ptmodule = PyTorchModule()
            res_contents = ptmodule.get_params()
        except Exception:
            self._set_headers_failed()
            return
        self._set_headers_success()
        response = {
            'token': "",
            'content': res_contents
        }
        self.wfile.write(bytes(json.dumps(response),'utf-8'))



    def do_OPTIONS(self):
        pass


    def do_POST(self):
        content_len = int(self.headers.get('content-length'))
        contents = self.rfile.read(content_len).decode('utf-8')
        try:
            dic = json.loads(contents)
            res_contents = []
            if dic['opcode'] == 'tsne':
                try:
                    imgs, latents, tsnes, labels = ptmodule.tsne_visualization()
                    for i in range(len(imgs)):
                        img = Image.fromarray(imgs[i]*255).convert('RGB')
                        res_contents.append({
                            'latent': latents[i].tolist(),
                            'img': self._img_to_base64(img),
                            'tsne_pos': tsnes[i].tolist(),
                            'label': int(labels[i])
                        })
                except Exception as err:
                    self._set_headers_failed()
                    print(err)
                    return
            elif dic['opcode'] == 'latent_imgs':
                try:
                    res_contents = {}
                    res_contents['tile']=[]
                    res_contents['linear']=[]
                    res_contents['target']=None
                    for i in range(len(dic['content'][0]['latent'])):
                        dic['content'][0]['latent'][i] = float(dic['content'][0]['latent'][i])
                    for i in [0,1]:
                        dic['content'][0]['target_idx'][i] = int(dic['content'][0]['target_idx'][i])
                    if not ptmodule.set_param('target_idx',dic['content'][0]['target_idx']):
                        raise Exception
                    res_contents = self._gen_imgs(dic['content'][0]['latent'], res_contents)
                except Exception as err:
                    self._set_headers_failed()
                    print(err)
                    return
            elif dic['opcode'] == 'encode_img':
                try:
                    res_contents = {}
                    res_contents['tile']=[]
                    res_contents['linear']=[]
                    res_contents['target']=None
                    img = Image.open(BytesIO(base64.b64decode(dic['content'][0]['img'].split(',')[-1])))
                    n_channels, im_size = ptmodule.get_im_status()
                    img = img.resize((im_size[0],im_size[1]))
                    if n_channels == 1:
                        img = np.array(img.convert('L'),dtype=np.float32)/255
                    latent = ptmodule.enc_img(img)
                    res_contents = self._gen_imgs(latent, res_contents)
                except Exception as err:
                    self._set_headers_failed()
                    print(err)
                    return
            elif dic['opcode'] == 'min_max':
                try:
                    mins, maxs = ptmodule.get_min_max()
                    if mins is None or maxs is None:
                        self._set_headers_failed()
                        return
                    else:
                        res_contents.append({
                            'n_dim': len(mins),
                            'min': mins,
                            'max': maxs
                        })
                except Exception as err:
                    self._set_headers_failed()
                    print(err)
                    return
            elif dic['opcode'] == 'set_param':
                try:
                    param_name = dic['content'][0]['param_name']
                    value = dic['content'][0]['value']
                    if not ptmodule.set_param(param_name,value):
                        self._set_headers_failed()
                        return
                except Exception as err:
                    self._set_headers_failed()
                    print(err)
                    return
            else:
                self._set_headers_not_found()
                return
            self._set_headers_success()
            response = {
                'opcode': dic['opcode'],
                'content': res_contents
            }
            self.wfile.write(bytes(json.dumps(response),'utf-8'))
        except Exception as err:
            self._set_headers_failed()
            print(err)
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-host", type=str, default=None)
    parser.add_argument("-port", type=int, default=None)
    args = parser.parse_args()

    ptmodule = None

    webServer = HTTPServer((args.host,args.port), DLModelServer)
    print("host: {}\t port: {}".format(args.host,args.port))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass
    webServer.server_close()

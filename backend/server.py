from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import argparse
from PIL import Image
from io import BytesIO
import base64
from DL import *
import time


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


    def do_GET(self):
        pass


    def do_OPTIONS(self):
        pass


    def do_POST(self):
        content_len = int(self.headers.get('content-length'))
        contents = self.rfile.read(content_len).decode('utf-8')
        try:
            dic = json.loads(contents)
            res_contents = []
            if dic['opcode'] == 'tsne':
                imgs, latents, tsnes, ys = ptmodule.tsne_visualization(100)
                for i in range(len(imgs)):
                    img = Image.fromarray(imgs[i]*255).convert('RGB')
                    res_contents.append({
                        'latent': latents[i].tolist(),
                        'img': self._img_to_base64(img),
                        'tsne_pos': tsnes[i].tolist(),
                        'label': int(ys[i])
                    })
            elif dic['opcode'] == 'latent_imgs': # TSNE visualization first, O.W. returns 404
                try:
                    for i in range(len(dic['content'][0]['latent'])):
                        dic['content'][0]['latent'][i] = float(dic['content'][0]['latent'][i])
                    for i in [0,1]:
                        dic['content'][0]['target_idx'][i] = int(dic['content'][0]['target_idx'][i])
                except Exception:
                    self._set_headers_failed()
                    return
                imgs, latents = ptmodule.latent_imgs_gen(dic['content'][0]['latent'],dic['content'][0]['target_idx'])
                if imgs is None or latents is None:
                    self._set_headers_failed()
                    return
                for i in range(len(imgs)):
                    img = Image.fromarray(imgs[i]*255).convert('RGB')
                    res_contents.append({
                        'latent': latents[i].tolist(),
                        'img': self._img_to_base64(img)
                    })
            elif dic['opcode'] == 'min_max': # TSNE visualization first, O.W. returns 404
                mins, maxs = ptmodule.get_min_max()
                if mins is None or maxs is None:
                    self._set_headers_failed()
                    return
                else:
                    res_contents.append({
                        'min': mins,
                        'max': maxs
                    })
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
    parser.add_argument("-host", type=str, default='147.46.215.63')
    parser.add_argument("-port", type=int, default=37373)
    args = parser.parse_args()

    ptmodule = PyTorchModule('MNIST')

    webServer = HTTPServer((args.host,args.port), DLModelServer)
    print("host: {}\t port: {}".format(args.host,args.port))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass
    webServer.server_close()

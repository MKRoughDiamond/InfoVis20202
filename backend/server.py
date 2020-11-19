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
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin','*')
        self.end_headers()


    def _set_headers(self,type_):
        self.send_response(200,"ok")
        self.send_header('Content-type', type_)
        self.send_header('Access-Control-Allow-Origin','*')
        self.end_headers()


    def do_GET(self):
        self._set_headers('application/json')
        self.wfile.write(bytes(json.dumps({}),'utf-8'))


    def do_OPTIONS(self):
        pass


    def do_POST(self):
        self._set_headers('application/json')
        content_len = int(self.headers.get('content-length'))
        contents = self.rfile.read(content_len).decode('utf-8')
        try:
            dic = json.loads(contents)
            res_contents = []
            if dic['opcode'] == 'tsne':
                imgs, latents, tsnes, ys = ptmodule.tsne_visualization(100)
                for i in range(len(imgs)):
                    img = Image.fromarray(imgs[i]).convert('RGB')
                    res_contents.append({
                        'latent': latents[i].tolist(),
                        'img': self._img_to_base64(img),
                        'tsne_pos': tsnes[i].tolist(),
                        'label': int(ys[i])
                    })
            elif dic['opcode'] == 'latent_imgs':
                # TODO: Select n points with given latent and target dimensions
                # TODO: Reconstruction
                pass
            else:
                self._set_headers_failed()
                return
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

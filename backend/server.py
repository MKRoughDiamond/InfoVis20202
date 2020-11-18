from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import argparse
from PIL import Image
from io import BytesIO
import base64


class DLModelServer(BaseHTTPRequestHandler):
    def _img_to_base64 (self,img):
        with BytesIO() as buf:
            img.save(buf, 'jpeg')
            imgbyte = buf.getvalue()
        return base64.b64encode(imgbyte).decode()


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
            if dic['opcode'] == 'print':
                self.wfile.write(bytes(json.dumps(dic['content']),'utf-8'))
            elif dic['opcode'] == 'img':
                im=Image.open("cat.jpg")
                res_contents = {'type':'process1','img':self._img_to_base64(im)}
                self.wfile.write(bytes(json.dumps(res_contents),'utf-8'))
            else:
                self.wfile.write(bytes(str(['print','img'])+'\n','utf-8'))
        except Exception as err:
            print(err)
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-host", type=str, default='147.46.215.63')
    parser.add_argument("-port", type=int, default=37373)
    args = parser.parse_args()
    webServer = HTTPServer((args.host,args.port), DLModelServer)
    print("host: {}\t port: {}".format(args.host,args.port))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass
    webServer.server_close()

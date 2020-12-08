# High-Dimensional Latent Space Visualization

![overview](overview.PNG)

### Environment (Recommended)
- Frontend
  - d3.js
  - canvas-free-drawing.js [(Link)](https://github.com/federico-moretti/canvas-free-drawing)
- Backend
  - Python == 3.6.10
  - torch == 1.5.0
  - torchvision == 0.6.0
  - Pillow == 7.1.2
  - numpy == 1.18.5
  - sklearn == 0.23.1
  - argparse

### Installing & Settings
#### Backend
- Install pretrained model
```
git submodule update --init --recursive
git clone --recursive https://github.com/MKRoughdiamond/InfoVis20202.git
pip install -r backend/model/requirements.txt
```
  - Backend server setting
```
python3 server.py -host (HOST_IP) -port (PORT)
```

#### Frontend
- Fix config.js to connect backend
```
cp config_example.js config.js
vi config.js
```
  - `config.js`
```javascript
const host = '(BACKEND_IP)';
const port = '(BACKEND_PORT)';
```
- Frontend server setting
```
python3 -m http.server (FRONTEND_PORT)
```

### Temp

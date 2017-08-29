# Yolo_v2_to_CoreML
This is a work in progress implementation of darknet YOLO to CoreML models converter.

The converter currently only supports Tiny YOLO_v2 style models and reference model. (route and reorg layers are not supported yet)

### Installation
```bash
virtualenv -p /usr/local/bin/python yoloml
source yoloml/bin/activate
pip install coremltools
pip install configparser
pip install numpy
```


## Converting the models

- Download Tiny YOLO cfg and weights from the [official YOLO website](http://pjreddie.com/darknet/yolo/).
- Run yolo_to_coreml.py
```bash
python yolo_to_coreml.py tiny-yolo-voc.cfg tiny-yolo-voc.weights tiny.mlmodel
```



# scale-aware-SSD

Training goes as follows:
  1. Collect hyper parameters from configuration file
        1. data parameters
                1. batch size
                2. train, test path
                3. val split percentage ( from training set )
                4. input size
                5. input format
                6. transforms
        2. learning parameters
                1. class weights
                2. detection 
                        1. bbox and cls lamda
                        2. loss function
                        3. minibatch size
                        4. samples choice ( dns, base ) 
        3. optimizer
                1. learning rate
                2. step
                3. epochs learning rate change
                4. wd lamda
        3. model parameters
                1. anchors ( aspects & scales )
                2. model type ( custom ssd )
                3. #classes 
        4. training parameters
                1. #epochs
  2. Create SSD model
        1. Generate Feauture Maps
        2. Initialize model
        3. Inialize bbox encoder
        4. Inialize bbox decoder
        5. Inialize bbox nms
  3. Apply bounding box encoder. 
        1. Resize bounding boxes to model's required shape.
  4. Initialize Loss Detection Object 
  5. Initialize Detection Engine
  6. Initialize Pytorch Lighting 
  7. Initialize Trainer Object
  8. Initialize Data Module
  9. Start Training with Training set
  10. Test Model wih Test set
  11. Run Coco Evaluator to acquire metrics

**Run:** python3 -m src.scripts.train -c src/configs/tod_config.json

**NVIDIA-SMI:** 460.39       
**Driver Version:** 460.39       
**CUDA Version:** 11.2
**Python:** 3.8.5

**Package                Version**
---------------------- -------------------
albumentations         0.5.2
conda                  4.9.2
detectron2             0.4+cu110
imageio                2.9.0
imageio-ffmpeg         0.4.3
imgaug                 0.4.0
ipython                7.19.0
matplotlib             3.4.1
moviepy                1.0.3
numpy                  1.20.2
opencv-python          4.5.1.48
pandas                 1.2.3
Pillow                 8.1.0
pip                    21.0.1
pycocotools            2.0.2
pytorch-lightning      1.2.6
PyYAML                 5.3.1
scikit-image           0.18.1
scikit-learn           0.22.2
scipy                  1.6.2
seaborn                0.11.1
simplejson             3.17.2
tensorboard            2.4.1
torch                  1.8.1+cu111
torchaudio             0.8.1
torchelastic           0.2.1
torchmetrics           0.2.0
torchvision            0.9.1+cu111
wheel                  0.35.1

**Dockerfile**
    
    FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

    ARG DEBIAN_FRONTEND=noninteractive

    RUN apt-get update --fix-missing && apt-get install -qy --upgrade \
      cmake \
      python3 python3-pip python3-setuptools \
      wget \
      libsm6 libxext6 libxrender-dev \
      xauth libgl1-mesa-glx \
      vim


    RUN pip3 install --upgrade pip 

    RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

    RUN pip3 install pytorch-lightning

    RUN pip3 install opencv-python

    RUN pip3 install matplotlib

    RUN pip3 install scikit-learn

    RUN pip3 install albumentations

    RUN pip3 install pycocotools

    RUN mkdir /src

    WORKDIR /src/

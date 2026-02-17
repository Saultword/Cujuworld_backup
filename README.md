# Cuju World - Football Style Recognition and Transfer Platform



Training based on [Pytorch](https://github.com/pytorch/pytorch)

Cuju World is an innovative platform that transforms sports moments into iconic masterpieces using advanced AI technology. The application analyzes football videos to recognize which professional player's style matches the user's technique, then allows users to generate personalized postcards by transferring their style to famous players like Messi or Ronaldo.

## Key Features

- **Player Style Recognition**: Upload your football video and discover which professional player your style resembles most
- **AI-Powered Postcard Generation**: Create personalized football postcards by transferring your style to famous players
- **Highlight Extraction**: Automatically identifies key moments from your videos
- **Historical Tracking**: View and manage your past analyses and generated content
- **Team Jersey Customization**: See yourself wearing the jersey of top football clubs

## Technology Stack

### Backend

- **Flask**: Python web framework for API development
- **MMA Action Recognition**: Pre-trained TSN model for football action analysis
- **Google Gemini API**: Style transfer and image generation
- **OpenCV**: Video processing and keyframe extraction
- **FFmpeg**: Video segmentation for detailed analysis

### Frontend

- HTML5/CSS3: Responsive and modern UI
- JavaScript: Interactive elements and API communication
- HTML5UP Template: Professional design foundation

## API Endpoints

| Endpoint                      | Method | Description                                            |
| :---------------------------- | :----- | :----------------------------------------------------- |
| `/analyze_video`              | POST   | Analyze football video and identify player style match |
| `/generate_style_transfer`    | POST   | Generate style-transferred image                       |
| `/get_keyframe_history`       | GET    | Retrieve historical keyframes                          |
| `/get_recent_style_transfers` | GET    | Get recently generated style transfers                 |
| `/keyframes/<filename>`       | GET    | Access stored keyframe images                          |

## Usage Guide

1. **Style Recognition** (`generic.html`):
   - Upload your football video
   - View analysis showing which professional player your style matches
   - See segmented analysis of different moments in your video
2. **Postcard Generation** (`elements.html`):
   - Select a keyframe from your history
   - Generate a style-transferred image with your favorite team's jersey
   - Download or share your personalized football postcard
3. **Homepage** (`index.html`):
   - View recent style transfers
   - Learn about the platform's capabilities
   - Navigate to different sections

## Customization

To customize the application:

1. **Team Jerseys**: Add new jersey images to `website_test/images/` and update `TEAM_JERSEY_MAP` in the backend
2. **Player Styles**: Modify `action_label.txt` to add new player styles
3. **UI Customization**: Edit the HTML/CSS files in `website_test/` directory

## Virtual Environment Setup (Anaconda) - Windows/Linux



| Spec             |              |
| ---------------- | ------------ |
| Operating System | Windows 11   |
| GPU              | None/ Nvidia |
| PyTorch version: | 1.13.1+cu117 |
| CUDA             | 11.7         |

### Step 1：Install Anaconda

https://docs.anaconda.com/anaconda/install/

### Step 2：Build a virtual environment

Run the following commands in sequence in Anaconda Prompt:

```
# 创建新环境
conda create --name pytorch117 python=3.9 -y

# 激活环境
conda activate pytorch117

# 安装PyTorch 1.13.1 + CUDA 11.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

#IF ERROR IN WINDOWS
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

Run the following command in the notebook or just conda install the package:

```
!pip install mmengine
!pip install mmcv==2.0.0rc4
!pip install mmdet
!pip install mmpose
! pip install mmaction2
! pip install numpy==1.26.4

```

Then download mmaction2 

[MMAction](https://github.com/open-mmlab/mmaction2)2

```
#only run on first time
git clone https://github.com/open-mmlab/mmaction2.git  
cd mmaction2
pip install -v -e .
```

And then clone the repository，copy the  website_test folder to the mmaction2 folder

## Documentation

### Training  Networks

You can prepare the training/verification video and labels document to train the network.

```
cuju-world/
├─mmaction2/  
├── website_test/                 # Frontend files
│   ├── assets/                   # CSS, JS, and images
│   ├── elements.html             # Postcard generation page
│   ├── generic.html              # Style recognition page
│   └── index.html                # Homepage
├── backend_test250630.py         # Main backend application
├── configs/                      # Model configuration files
│   └── recognition/
├── data/                         # Label data
│   └── action_label.txt
├── work_dirs/                    # Pre-trained models
│   └── new_backend_model.pth
├── requirements.txt              # Python dependencies
└── README.md                    # This documentation
```

Use `train.py` to train a new  network. Run `python train.py` to view all the possible parameters.  Example usage:

### Prepare training videos

You can prepare your training videos follow the guidance on mmaction2 or download the prepared 

copy the training video to mmaction2/data/train

and val videos to mmaction2/data/val.

Then classify the video files' represented action styles using the following format and output to a .txt file

train/val_label.txt：

```
bz_kick_train22.mp4 1  #class1
MS_kick_train55.mp4 2 #class2
bz_kick_train20.mp4 1
mod_kick30_flip.mp4 3#class3
```

Structure:

```
cuju-world/
├─mmaction2/  
├── data/                
│   ├── train/                
│    ├── video1.mp4            
│    ├── video2.mp4           
│    └── train_label.txt             
│   ├── val/                
│    ├── video1.mp4            
│    ├── video2.mp4           
│    └── val_label.txt               
```

Or just download the videos with labels on Google bucket:

```
gsutil -m cp -r gs://testing_soccer1/soccer_testset_541_8_2_104 mmaction2/data/val
gsutil -m cp -r gs://testing_soccer1/testing_soccer1/soccer_train_541_8_2_437 mmaction2/data/train 
```

And you can run the autolabel.ipynb to generate the label.txt for different floder

Then train the model based on TSN(Sample) or other models on mmaction2 you find

### Run on training config

    python mmaction2/train.py mmaction2/configs/recognition/tsn/tsn_20250513_480_tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb-Copy1.py --work-dir  mmaction2/work_dirs

### Website backend 

Run the backend by

```
cd mmaction2
python backend_test250630.py
```

### Website pages

The website run on https://localhost:5000 in default


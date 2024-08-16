# [ICIP 2023] Compound Multi-branch Feature Fusion for Image Deraindrop  
## [Chi-Mao Fan](https://github.com/FanChiMao), Tsung-Jung Liu, Kuan-Hsien Liu   
[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/abs/2206.02748v1) 
[![official_paper](https://img.shields.io/badge/IEEE-Paper-blue)](https://ieeexplore.ieee.org/document/10222907) 
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)](https://youtu.be/8UCxeuP2A_Q?si=Iid5GiaqqdEBnAZl) 
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://docs.google.com/presentation/d/1h4Y2CMxx7j6V72kWG5fD_S9Tzn1Vzguh/edit?usp=drive_link&ouid=108348190349543369603&rtpof=true&sd=true) 
[![poster](https://img.shields.io/badge/Summary-Poster-9cf)](https://drive.google.com/file/d/1lMF-SqvjlZ4frQI9REfRMRU8n0Tx1b1C/view?usp=sharing) 
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/52Hz/CMFNet_deblurring) 
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FFanChiMao%2FCMFNet&label=visitors&countColor=%232ccce4&style=plastic)  

> Abstract : Image restoration is a challenging and ill-posed problem which also has been a long-standing issue. However, most of learning based restoration methods are proposed to target one degradation type which means they are lack of generalization. In this paper, we proposed a multi-branch restoration model inspired from the Human Visual System (i.e., Retinal Ganglion Cells) which can achieve multiple restoration tasks in a general framework. The experiments show that the proposed multi-branch architecture, called CMFNet, has competitive performance results on four datasets, including image deblurring, dehazing and deraindrop which are very common applications for autonomous cars.

## Network Architecture  

<table>
  <tr>
    <td colspan="2"><img src = "https://i.ibb.co/3WRbpYv/CMFNet.png" alt="CMFNet" width="800"> </td>  
  </tr>
  <tr>
    <td colspan="2"><p align="center"><b>Overall Framework of CMFNet</b></p></td>
  </tr>
  
  <tr>
    <td> <img src = "https://i.ibb.co/FBx9QLy/UNet.png" width="400"> </td>
    <td> <img src = "https://i.ibb.co/W0yk5hn/MSC.png" width="400"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Branch U-Net architecture</b></p></td>
    <td><p align="center"> <b>Mixed Skip Connection (MSC)</b></p></td>
  </tr>
</table>


## Quick Run  
You can simply demo on my space of [**Hugging Face**](https://huggingface.co/52Hz)  
- [**Dehazing**](https://huggingface.co/spaces/52Hz/CMFNet_dehazing)  
- [**Deraindrop**](https://huggingface.co/spaces/52Hz/CMFNet_deraindrop)  
- [**Deblurring**](https://huggingface.co/spaces/52Hz/CMFNet_deblurring)  

Or test on local environment:  
To test the pre-trained models of Deraindrop, Dehaze, Deblurring on your own images, run
```
python demo.py --input_dir images_folder_path --result_dir save_images_here --weights path_to_models
```
Here is an example to perform Deraindrop:
```
python demo.py --input_dir './demo_samples/deraindrop' --result_dir './demo_results' --weights './pretrained_model/deraindrop_model.pth'
```
**All pre-trained models can be downloaded at [pretrained_model/README.md](pretrained_model/README.md) or [here](https://github.com/FanChiMao/CMFNet/releases)**
## Train  
To train the restoration models of Deraindrop, Dehaze and Deblurring. You should check the following components:  
- `training.yaml`:  
  ```
  # Training configuration
  GPU: [0,1,2,3]

  VERBOSE: False

  MODEL:
    MODE: 'Deblur'

  # Optimization arguments.
  OPTIM:
    BATCH: 2
    EPOCHS: 150
    # NEPOCH_DECAY: [10]
    LR_INITIAL: 2e-4
    LR_MIN: 1e-6
    # BETA1: 0.9

  TRAINING:
    VAL_AFTER_EVERY: 1
    RESUME: False
    TRAIN_PS: 256
    VAL_PS: 256
    TRAIN_DIR: './datasets/deraindrop/train'       # path to training data
    VAL_DIR: './datasets/deraindrop/test' # path to validation data
    SAVE_DIR: './checkpoints'           # path to save models and images

  ```
  
- Details of Hyperparameters
	```
	-------------------------------------------------
	GoPro dataset:
	Training patches: 33648 (2103 x 16)
	Validation: 1111
	Initial learning rate: 2e-4
	Final learning rate: 1e-6
	Training epochs: 150 (120 is enough)
  Training time (on single 2080ti): about 10 days

	Raindrop dataset:
	Training patches: 6888 (861 x 8)
	Validation: 1228 (307 x 4)
	Initial learning rate: 2e-4
	Final learning rate: 1e-6
	Training epochs: 150 (100 is enough)
  Training time (on single 1080ti): about 2.5 days
  
	IO-Haze dataset:
	Training patches: 15000 (75 x 200)
	Validation: 55
	Initial learning rate: 1e-4
	Final learning rate: 1e-6
	Training epochs: 150 (50 is enough)
  Training time (on single 1080ti): about 3 days
	-------------------------------------------------
	```  
  
- Dataset:  
  The preparation of dataset in more detail, see [datasets/README.md](datasets/README.md).  
  
- Train:  
  If the above path and data are all correctly setting, just simply run:  
  ```
  python train.py
  ```  

## Test (Evaluation)  
To test the models of Deraindrop, Dehaze, Deblurring with ground truth, see the `test.py` and run  
```
python test.py --input_dir images_folder_path --result_dir save_images_here --weights path_to_models --dataset type_of_task --gpus CUDA_VISIBLE_DEVICES
```
Here is an example to perform Deraindrop:  
```
python test.py --input_dir './datasets/' --result_dir './test_results/' --weights './pretrained_model/deraindrop_model.pth' --dataset deraindrop --gpus '0'
```  
To test the PSNR and SSIM of *Deraindrop*, see the `evaluation_Y.py` and run  
```
python evaluation_Y.py --input_dir path_to_restored_images --gt_dir path_to_gt_images
```
Here is an example:  
```
python valuation_Y.py --input_dir './test_results/deraindrop' --gt_dir './demo_samples/deraindrop'
```  
And to test the PSNR and SSIM of *Dehaze* and *Deblur*, see the `evaluation_RGB.m`  

## Results
<details>  
<summary>Result Tables (Click to expand)</summary>  

  | Restoration task |     Result Tables    |
  | :--------------: | :------------------: |
  | Deraindrop       |<img src = "https://i.ibb.co/MMMj75H/deraindrop-table.png" width="500">|
  | Dehaze           |<img src = "https://i.ibb.co/wRY9Sr6/dehaze-table.png" width="500">|
  | Deblur           |<img src = "https://i.ibb.co/xFhGCY0/deblur-table.png" width="620">|

</details>  

## Visual Comparison  
<details>  
<summary>Visual Comparison Figures (Click to expand)</summary>  

  | Restoration task |    Restored images   |  Ground Truth     |
  | :--------------: | :------------------: | :---------------: |
  | Deraindrop       |<img src="figures/deraindrop_bf.gif" alt="deraindrop_bf" width="300" style="zoom:100%;" />|<img src="https://i.ibb.co/L5KxZSP/105-clean.jpg" alt="deraindrop_gt" width="300" style="zoom:100%;" />|
  | Dehaze           |<img src="figures/dehaze_bf.gif" alt="dehaze_bf.gif" width="300" style="zoom:100%;" />|<img src="https://i.ibb.co/7Q5BKZS/47-gt.png" alt="dehaze_gt.png" width="300" style="zoom:100%;" />|  
  | Deblur           |<img src="figures/deblur_bf.gif" alt="deblur_bf.gif" width="300" style="zoom:100%;" />|<img src="https://i.ibb.co/yf6d5XG/GOPR0384-11-00-000001.png" alt="https://i.ibb.co/1JrwL1Z/deblur-gt.png" width="300" style="zoom:100%;" />|

</details>  

**More visual results can be downloaded at [here](https://github.com/FanChiMao/CMFNet/releases).**  

## Citation  
```
@inproceedings{fan2023compound,
  title={Compound Multi-Branch Feature Fusion for Image Deraindrop},
  author={Fan, Chi-Mao and Liu, Tsung-Jung and Liu, Kuan-Hsien},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)},
  pages={3399--3403},
  year={2023},
  organization={IEEE}
}

```

## Contact
If you have any question, feel free to contact qaz5517359@gmail.com  



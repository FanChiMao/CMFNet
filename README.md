# Compound Multi-branch Feature Fusion for Real Image Restoration  
## [Chi-Mao Fan](https://github.com/FanChiMao), Tsung-Jung Liu, Kuan-Hsien Liu  

**Paper**:  https://paperswithcode.com/paper/compound-multi-branch-feature-fusion-for-real  

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
If you use CMFNet, please consider citing:  

## Contact


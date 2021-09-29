# Compound Multi-branch Feature Fusion for Real Image Restoration (ICLR 2022)  
## [Chi-Mao Fan](https://github.com/FanChiMao), Tsung-Jung Liu, Kuan-Hsien Liu  

Paper:  

Video Presentation:  

Presentation Slides:  

***
> Abstract : Image restoration is a challenging and ill-posed problem which also has been a long-standing issue. However, most of learning based restoration methods are proposed to target one degradation type which means they are lack of generalization. In this paper, we proposed a multi-branch restoration model inspired from the Human Visual System (i.e., Retinal Ganglion Cells) which can achieve multiple restoration tasks in a general framework. The experiments show that the proposed multi-branch architecture, called CMFNet, has competitive performance results on four datasets, including image deblurring, dehazing and deraindrop which are very common applications for autonomous cars.

## Network Architecture  
<img src = "./figures/CMFNet.png" width="750">  

## Quick Run  
To test the pre-trained models of Deraindrop, Dehaze, Deblurring on your own images, run
```
python demo.py --input_dir images_folder_path --result_dir save_images_here --weights path_to_models
```
Here is an example to perform Deraindrop:
```
python demo.py --input_dir ./demo_samples/deraindrop --result_dir ./demo_results --weights ./pretrained_model/deraindrop_model.pth
```
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
  The preparation of dataset in more detail, see `datasets/README.md`.
- Train:  


## Test  

## Results
<details>  
<summary>Result Tables: </summary>  

  | Restoration task |     Result Tables    |
  | :--------------: | :------------------: |
  | Deraindrop       |<img src = "./figures/deraindrop_table.png" width="500">|
  | Dehaze           |<img src = "./figures/dehaze_table.png" width="500">|
  | Deblur           |<img src = "./figures/deblur_table.png" width="620">|
  
</details>  

## Visual Comparison  
<details>  
<summary>Visual Comparison Figures: </summary>  
  
  | Restoration task |    Restored images   |  Ground Truth     |
  | :--------------: | :------------------: | :---------------: |
  | Deraindrop       |<img src="figures/deraindrop_bf.gif" alt="deraindrop_bf" width="300" style="zoom:100%;" />|<img src="figures/deraindrop_gt.jpg" alt="deraindrop_gt" width="300" style="zoom:100%;" />|
  | Dehaze           |<img src="figures/dehaze_bf.gif" alt="dehaze_bf.gif" width="300" style="zoom:100%;" />|<img src="figures/dehaze_gt.png" alt="dehaze_gt.png" width="300" style="zoom:100%;" />|  
  | Dehaze           |<img src="figures/deblur_bf.gif" alt="deblur_bf.gif" width="300" style="zoom:100%;" />|<img src="figures/deblur_gt.png" alt="deblur_gt.png" width="300" style="zoom:100%;" />|

</details>  

## Citation  
If you use CMFNet, please consider citing:  
```
@inproceedings{,
    title={},
    author={Chi-Mao Fan, Tsung-Jung Liu, Kuan-Hsien Liu},
    booktitle={ICLR},
    year={2022}
}
```

## Contact
If you have any question, please contact qaz5517359@gmail.com  

# Compound Multi-branch Feature Fusion for Real Image Restoration (ICLR 2022)  
## [Chi-Mao Fan](https://github.com/FanChiMao), Tsung-Jung Liu, Kuan-Hsien Liu  

Paper:  

Video Presentation:  

Presentation Slides:  

***
> Abstract : Image restoration is a challenging ill-posed problem which also has been a long-standing issue. 
> However, most of learning based restoration methods are proposed to solve one degradation type which means they are lack of the generalization.
> In this paper, we proposed a multi-branch restoration model inspired from the Human Visual System of Retina Ganglion Cells which support well with multiple restoration tasks in a general framework.
>  Our experiments show that the proposed multi-branch architecture, called CMFNet, performs competitive results on seven datasets include image deblurring, dehazing and deraindrop which are the common degradations on autonomous cars' camera.

## Network Architecture  
<img src = "./figures/CMFNet.png" width="750">  

## Quick Run  
To test the pre-trained models of Deraindrop, Dehaze, Deblurring on your own images, run
```
python demo.py --task Task_Name --input_dir path_to_images --result_dir save_images_here --weights path_to_models
```

## Results
<details>  
<summary>Performance table: </summary>  

- Deraindrop  
  <img src = "./figures/deraindrop_table.png" width="500">

- Dehaze  
  <img src = "./figures/dehaze_table.png" width="500">

- Deblur  
  <img src = "./figures/deblur_table.png" width="750">

</details>  
  
## Contact
If you have any question, please contact qaz5517359@gmail.com  

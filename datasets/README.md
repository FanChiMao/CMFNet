# Datasets  
## Deraindrop  
- DeRainDrop(train & test): 
  - [train & test](https://drive.google.com/open?id=1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K  )
## Dehaze  
- I-Haze(train):  
  - [train](https://data.vision.ee.ethz.ch/cvl/ntire18/i-haze/)  
- O-Haze(train):  
  - [train](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/)  
- Dense-haze(test):  
  - [test](https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/)  
- NH-haze(test):  
  - [test](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/)  

## Deblur  
- GoPro(train & test):  
  - [train](https://drive.google.com/drive/folders/1AsgIP9_X0bg0olu2-1N6karm2x15cJWE)  
  - [test](https://drive.google.com/drive/folders/1a2qKfXWpNuTGOm2-Jex8kfNSzYJLbqkf)  
- HIDE(test):  
  - [test](https://drive.google.com/drive/folders/1nRsTXj4iTUkTvBhTcGg8cySK8nd3vlhK)  


## Preprocess  
You have to crop the high-resolution training images to fixed size training patches by [`generate_patches.py`](./generate_patches.py).  
Our experiment setting can refer `Details of Hyperparameters` of train section in main README.  

## Tree  
  ```
  datasets
    ├── deraindrop  
    |    ├── test
    |    |     ├── input
    |    |     └── target    
    |    └── train
    |          ├── input
    |          └── target    
    ├── dehaze
    |    ├── test
    |    |     ├── input
    |    |     └── target    
    |    └── train
    |          ├── input
    |          └── target    
    └── deblur
         ├── test
         |     ├── input
         |     └── target    
         └── train
               ├── input
               └── target    


  ```  

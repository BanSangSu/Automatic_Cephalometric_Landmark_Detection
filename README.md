# Automatic Cephalometric Landmark Detection on X-ray Images
### Encouragement prize 🏆
- **Organiser**: National Program of Excellence in Software
- **Period**: 14th Jul, 2021


## Abstract
My **best** Automatic Cephalometric Landmark Detection model used **UNet++** (Using Pytorch). Backbone structure in UNet++ encoding layers was **VGG**.


## Datasets
You can see Cephalometric landmark datasets in [data](data). In train and val folders, the 0img folder contains cephalometric images. Each of the 101 through 110 folders contains heapmap landmark images. Every 10 heatmap landmarks represents a landmark in one cephalometric image. (e.g. train/0img/001.png <=> train/101/001.png ~ 010.png)


## Results
**Note**  
🟢 Green points are ground truth. 🔴 Red points are results of UNet++ Network_0.001~ (Batch size 4). 🔵Blue points are results of UNet++ Network_0.005~ (Batch size 1).


![image](https://github.com/BanSangSu/Automatic_Cephalometric_Landmark_Detection/assets/76412884/e2f4b232-0824-49cb-9daf-a05b39ab473d)  


The performance of the two models, trained with different batch sizes, closely aligns with the ground truth.


- You can see more details in [main.ipynb](main.ipynb).


## Feature
- **Model**: UNet++
- **Optimiser**: AdamW
- **Scheduler**: CosineAnnealingLR
- **Loss function**: L1Loss


## Experiments
- **Model**: **UNet++** & Unet

    **UNet++** model **converged** faster than Unet.  


- **Batch size**: **Batch size 1** & Batch size 4

    The results for the two batch sizes were **not significantly different**. But, the loss in batch size 4 was much **lower** than the loss in batch size 1. As the training continued with a **batch size of 4**, the training loss decreased, but the validation loss increased, indicating a potential **overfitting problem**.
    
    **Note**  
    If you have a GPU with **a large amount of memory** and **a lot of data**, consider **increasing the batch size**. It could **improve** the performance.
    

- **Loss function**: **L1Loss** & SmoothL1Loss

    The model using **SmoothL1Loss** had a very small loss, around **1e-6**, while the model using L1Loss had a loss of approximately 5e-3. Despite the smaller loss value, the performance of the model with **SmoothL1Loss** was **significantly worse** than that of the model with L1Loss.


## Summary
By changing **the loss function**, I saw a dramatic **improvement** in performance. It's essential to examine **the actual output for proper training**, and you should compare the loss to this output. The loss that corresponds to the actual output can serve as a metric to evaluate the results.


## License
[BSD 3-Clause License](LICENSE)
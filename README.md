# [Re] CRATE (Coding RAte reduction TransformEr)
This repository provides a reproducibility implementation of the paper in PyTorch.

- **White-Box Transformers via Sparse Rate Reduction** [**NeurIPS-2023**, [paper link](https://openreview.net/forum?id=THfl8hdVxH#)]. By [Yaodong Yu](https://yaodongyu.github.io) (UC Berkeley), [Sam Buchanan](https://sdbuchanan.com) (TTIC), [Druv Pai](https://druvpai.github.io) (UC Berkeley), [Tianzhe Chu](https://tianzhechu.com/) (UC Berkeley), [Ziyang Wu](https://robinwu218.github.io/) (UC Berkeley), [Shengbang Tong](https://tsb0601.github.io/petertongsb/) (UC Berkeley), [Benjamin D Haeffele](https://www.cis.jhu.edu/~haeffele/#about) (Johns Hopkins University), and [Yi Ma](http://people.eecs.berkeley.edu/~yima/) (UC Berkeley). 

## Dataset
We pretrained the model using ImageNet-100, downloading it using the kaggle API command
```python
kaggle datasets download -d ambityga/imagenet100
```
The included file folds need to be merged into train and val folders.

## Training
To train a CRATE model on ImageNet-1K, run the following script (training CRATE-tiny)

As an example, we use the following command for training CRATE-tiny on ImageNet-100:
```python
python main.py 
  --arch {model_name} 
  --batch-size 512 
  --epochs 200 
  --optimizer Lion 
  --lr 0.0002 
  --weight-decay 0.05 
  --print-freq 25 
  --data DATA_DIR
```
and replace `DATA_DIR` with `[imagenet-folder with train and val folders]`.


## Finetuning

```python
python finetune.py 
  --bs 256 
  --net {model_name}
  --opt adamW  
  --lr 5e-5 
  --n_epochs 200 
  --randomaug 1 
  --data {cifar10/cifar100/flower/pets}
  --ckpt_dir CKPT_DIR 
  --data_dir DATA_DIR
```
Replace `CKPT_DIR` with the path for the pretrained CRATE weight, and replace `DATA_DIR` with the path for the dataset. The `CKPT_DIR` could be `None`, the system will automatically check the data folder to verify its presence, and if absent, it will proceed to download it. 

## Demo: Emergent segmentation in CRATE

CRATE models exhibit emergent segmentation in their self-attention maps solely through supervised training.
The Colab Jupyter notebook visualize the emerged segmentations from a supervised **CRATE** model. The demo provides visualizations which match the segmentation figures above.

Link: [re-crate-emergence.ipynb](https://colab.research.google.com/drive/1sOv-VGFRGVo82rLq9QrmFodTP9E2Y5Nu?usp=sharing) (in colab)





```
@article{yu2024white,
  title={White-Box Transformers via Sparse Rate Reduction},
  author={Yu, Yaodong and Buchanan, Sam and Pai, Druv and Chu, Tianzhe and Wu, Ziyang and Tong, Shengbang and Haeffele, Benjamin and Ma, Yi},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

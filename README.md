# LSE-CDM
# Latent space Semantic Enhancement with Conditional
Diffusion Modeling

## Environment

- Anaconda 3
- python 3.8.10
- pytorch 1.12.0
- numpy 1.22.3

## Usage

#### LSE-CDM train

```
cd ./LSE-CDM
python main.py --cuda --dataset=$1 --data_path=../datasets/$1/ --emb_path=../datasets/ --lr1=$2 --lr2=$3 --wd1=$4 --wd2=$5 --batch_size=$6 --n_cate=$7 --in_dims=$8 --out_dims=$9 --lamda=${10} --mlp_dims=${11} --emb_size=${12} --mean_type=${13} --steps=${14} --noise_scale=${15} --noise_min=${16} --noise_max=${17} --sampling_steps=${18} --reweight=${19} --log_name=${20} --round=${21} --gpu=${22}
```
#### Data

The experimental data are in './LSE-CDM/datasets' folder. Datas are available at [here](https://drive.google.com/drive/folders/1V2OT3rnWPZNkTf3FAfK6De_iKyKVOUBb?usp=sharing). Please place the downloaded data in the dataset folder.

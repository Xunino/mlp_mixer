# MLP Mixer

This implementation from paper [MLP Mixer](https://arxiv.org/pdf/2105.01601.pdf). Give us a star if you like this repo.

## Architecture Image

<p align="center">
    <img src="https://github.com/Xunino/mlp_mixer/blob/main/assets/net.png">
</p>

Authors:

- Github: Xunino
- Email: ndlinh.ai@gmail.com

## I. Set up environment

- Step 1:

```bash
conda create -n {your_env_name} python==3.7.0
```

- Step 2:

```bash
conda env create -f environment.yml
```

- Step 3:

```bash
conda activate {your_env_name}
``` 

## II. Set up your dataset

- Guide user how to download your data and set the data pipeline

- Data pipeline example:

```
train/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
...class_c/
......c_image_1.jpg
......c_image_2.jpg
```

```
val/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
...class_c/
......c_image_1.jpg
......c_image_2.jpg
```

## III. Training process:

**Script training:**

```
python train.py --train-path={dataset/train} --val-path={dataset/val} --batch-size=32 --epochs=100 --n_blocks=8 --C=512 --DC=1024 --DS=256 --image-size=224 --patch-size=32 --augments=False --retrain=False
```
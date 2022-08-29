HoechstGAN
==========

This repository contains the code for the paper *HoechstGAN: Virtual Lymphocyte Staining Using Generative Adversarial
Networks*.

The structure of the code is inspired by the [pytorch implementation of pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), but has been heavily modified.
We use [hydra](https://hydra.cc) for configuration management and [wandb](http://wandb.ai) for tracking experiments.

![](img/hoechstgan_notex.svg)

## Installing
We provide a `Dockerfile` as well as a `docker-compose.yml` file that builds the Docker container and mounts the code (i.e. this repository) as a volume.
```
docker-compose up -d --build
```
Then, in the `hoechstgan` container, run
```
./install.sh
```
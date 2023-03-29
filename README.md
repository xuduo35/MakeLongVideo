## MakeLongVideo - Pytorch
Implementation of long video generation based on diffusion model.

## Setup
### Requirements

```shell
python3 -m pip install -r requirements.txt
```

## Training
### Prepare Stable Diffusion Pretrained Weights
download from huggingface and put it in directory 'checkpoints' which is configured in configs/makelongvideo.yaml 

### Download webvid dataset
download webvid dataset into directory 'data/webvid' using https://github.com/m-bain/webvid repo. Then prepare dataset using command
```shell
python3 genvideocap.py
```

### Begin training
accelerate launch --config_file ./configs/multigpu.yaml train.py --config configs/makelongvideo.yaml

## Inference
```shell
python3 infer.py  --width 128 --height 128 --prompt "a panda is surfing"
```

## Todo
- [ ] generate 24 frames video of 128x128
- [ ] add fps control
- [ ] release pretrained checkpoint
- [ ] improve resolution to 256x256, 512x512
- [ ] 1~2minutes video generation
- [ ] make story video

## References
* Make-A-Video: https://github.com/lucidrains/make-a-video-pytorch
* Tune-A-Video: https://github.com/showlab/Tune-A-Video
* diffusers: https://github.com/huggingface/diffusers

## Citations

```bibtex
@misc{Singer2022,
    author  = {Uriel Singer},
    url     = {https://makeavideo.studio/Make-A-Video.pdf}
}
```

```
@article{wu2022tuneavideo,
    title   = {Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation},
    author  = {Wu, Jay Zhangjie and Ge, Yixiao and Wang, Xintao and Lei, Stan Weixian and Gu, Yuchao and Hsu, Wynne and Shan, Ying and Qie, Xiaohu and Shou, Mike Zheng},
    journal={arXiv preprint arXiv:2212.11565},
    year    = {2022},
    note    = {under review}
}
```

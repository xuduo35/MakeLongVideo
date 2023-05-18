## MakeLongVideo - Pytorch
Implementation of long video generation based on diffusion model.

<table class="center">
  <tr>
  <td style="text-align:center;" width="170">"Ironman is surfing"</td>
  <td style="text-align:center;" width="170">"a car is racing"</td>
  <td style="text-align:center;" width="170">"a cat eating food of a bowl, in von Gogh style"</td>
  <td style="text-align:center;" width="170">"a giraffe underneath the microwave"</td>
  </tr><tr>
  <td><img src="samples/Ironman is surfi-7P269S.gif" width="170"></td>
  <td><img src="samples/a car is racing-tm9rwR-4x.gif" width="170"></td>
  <td><img src="samples/a cat eating foo-IR47a4.gif" width="170"></td>
  <td><img src="samples/a giraffe undern-cAmGAc.gif" width="170"></td>
  </tr><tr>
  <td style="text-align:center;" width="170">"a glass bead falling into water with huge splash"</td>
  <td style="text-align:center;" width="170">"a video of Earth rotating in space"</td>
  <td style="text-align:center;" width="170">"A teddy bear running in New York City"</td>
  <td style="text-align:center;" width="170">"A stunning aerial drone footage time lapse of El Capitan in Yosemite National Park at sunset"</td>
  </tr><tr>
  <td><img src="samples/a glass bead fal-Uxxg0y.gif" width="170"></td>
  <td><img src="samples/a video of Earth-DzP1ma.gif" width="170"></td>
  <td><img src="samples/A teddy bear run-A17vOA.gif" width="170"></td>
  <td><img src="samples/A stunning aeria-WdIUoM.gif" width="170"></td>
  </tr>
</table >

<!-- <br>   -->

## Setup
### Requirements

```shell
python3 -m pip install -r requirements.txt
```

## Training
### Prepare Stable Diffusion v1-4 pretrained weights
download from huggingface and put it in directory 'checkpoints' which is configured in configs/makelongvideo.yaml 

### Download webvid dataset
download webvid dataset into directory 'data/webvid' using https://github.com/m-bain/webvid repo. Then prepare dataset using command
```shell
python3 genvideocap.py
```

### Download LAION400M dataset
download laion400m into directory 'data/laion400m'

### Train
first train using resolution 128x128
```shell
accelerate launch --config_file ./configs/multigpu.yaml train.py --config configs/makelongvideo.yaml
```

then finetune in resolution 256x256, modify last line of configs/makelongvideo256x256.yaml according to your local epoch checkpoint
```shell
accelerate launch --config_file ./configs/multigpu.yaml train.py --config configs/makelongvideo256x256.yaml
```

## Inference
```shell
# unwrap checkpoint first
TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch train.py --config configs/makelongvideo.yaml --unwrap ./outputs/makelongvideo/checkpoint-3000
```

inference directly
```shell
python3 infer.py  --width 256 --height 256 --prompt "a panda is surfing"
```

inference using latents initialized by sample video
```shell
python3 infer.py  --width 256 --height 256 --prompt "a panda is surfing" --sample_video_path your_sample_video
```

inference by sample frame rate 6 (actual frame rate is 24/6==4)
```shell
python3 infer.py  --width 256 --height 256 --prompt "a panda is surfing" --speed 6
```

## Todo
- [x] generate 24 frames video of 256x256
- [x] add fps control
- [ ] release pretrained checkpoint
- [ ] improve resolution to 512x512
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

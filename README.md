# Stable-Diffusion-temperature-settings
Provides the ability to set the temperature for both UNET and CLIP. For ComfyUI.

I also added a togglable function compatible with SD 1.x models and SDXL/Turbo which helps to preserve quality weither it is for downscaling or upscaling.

## Usage

Like any other model patch:

![image](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/baafc52a-452b-499f-8aec-2092b019e71f)


## CLIP temperature at 0.75, 1 and 1.25. Prompt "a bio-organic living plant spaceship"

![combined_image_new](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/12034834-43d0-44a5-a603-6c87d1bc6e5d)


## Using SDXL, 512x512, 256x256, 128x128 without / with modification on the U-Net:

![335007111-bf6d7ef0-9c18-4436-8037-6b60a6a37ce2](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/e47775e0-1b36-46f9-9eac-5467ed8b6715)![335007043-0c4540ab-1840-4230-940a-07a9e38ef38a](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/3aecf1b1-85a8-4362-a7e1-e76048ca5f4b)

![335007132-3379081c-2c4e-4af0-ba92-b57031b3845b](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/ec1680c2-bb8f-4c50-855d-aeb5e0858a05)![335007062-a4dc0de9-68b7-4158-b4fa-6c607862d04a](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/1304717f-91d6-48a9-991f-b9a725ddff9d)


![335007140-10991fbe-4123-46d2-8069-cfaece9e77ec](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/b8e33680-cc4c-420c-b128-6b790a05ca12)![335007077-4d17e360-0e28-4fd2-98ae-8f6944114815](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/ff1c7ef5-1bbb-49bb-b53d-f14c4efe44bd)



## Using SD v1-5-pruned-emaonly:

- Lower temperature applied to all layers except input 1 and 2, output 9, 10 and 11.
- At 0.71.
- Only self-attention
- Resolution at 1024*512

![combined_pair_2](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/5e5403ea-2cb3-462c-a9f1-6cc7b1ddbaea)
![combined_pair_1](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/84fed1e4-a7ba-4f2a-8562-e3573f0aab8f)
![combined_pair_3](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/c6703c21-0d63-404e-9bf8-3a7c580f59e7)




# Patreon

Provide an incentive to contributors:

https://www.patreon.com/extraltodeus

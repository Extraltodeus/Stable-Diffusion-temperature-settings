# Stable-Diffusion-temperature-settings
Provides the ability to set the temperature for both UNET and CLIP. For ComfyUI.

## Usage

Like any other model patch:

![image](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/32b73433-df6a-4c49-99a6-5ddf21a4777a)


## CLIP temperature at 0.75, 1 and 1.25. Prompt "a bio-organic living plant spaceship"

![combined_image_new](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/12034834-43d0-44a5-a603-6c87d1bc6e5d)

It basically either refine or dilutes the precision.


## All the nodes

![image](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/2d04cb28-2a1d-4384-8e62-9e9a6b0829dc)

- Model specific nodes only have for difference a custom auto-scaling when set at 0

- The CLIP temperature patches the attention within the VRAM/RAM. The modification ignores connections but is only linked to what is loaded. To get default behavior set the temperature at one.


----
Most things below relates to the relation with resolution.
----

## Resolution unlocking side-effect:

Changing the UNET temperature allows to obtain better results a different resolutions.

It needs to be higher for smaller resolutions and lower for higher resolutions.

To this effect I wrote a dynamic scaling temperature function. It sets itself relatively to the selected resolution.

**TO USE IT**: Select the patch corresponding to your model (currently SD1.x or SDXL). Set the temperature slider at 0.

It is imperfect for resolutions higher than 1.5x as the amplification becomes too strong. But it is very effective for smaller. For higher resolutions you can always set the temperature manually or do a double-pass or use the "light" intensity.

There seem to be a proportional trend. I would however need faster hardware to be able to generate more samples and become able to detect it. I opened the discussions for this repository if you feel like sharing your best settings.


# Examples

### Using SDXL, 512x512, 256x256, 128x128 with/without:

![03263UI_00001_](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/0c4540ab-1840-4230-940a-07a9e38ef38a)![03276UI_00001_](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/a4dc0de9-68b7-4158-b4fa-6c607862d04a)![03287UI_00001_](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/4d17e360-0e28-4fd2-98ae-8f6944114815)



![03296UI_00001_](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/bf6d7ef0-9c18-4436-8037-6b60a6a37ce2)![03297UI_00001_](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/3379081c-2c4e-4af0-ba92-b57031b3845b)![03298UI_00001_](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/10991fbe-4123-46d2-8069-cfaece9e77ec)


### SDXL 1536x1536, single pass, 12 steps. I let you guess which is which.

![03302UI_00001_](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/5417058f-f5c7-4d8b-838f-a13962e6d85d)![03312UI_00001_](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/56144013-d85c-430d-a256-6f752aee4799)


### Here using SDXL at a resolution of 328x328. First row temperature at 1, second row using dynamic scaled attention:

![image_grid](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/7b5b2ffb-f621-4eca-9f97-04f78c2eaf7c)

### Non cherry-picked SD v1-5-pruned-emaonly at 512*1024, first row with the dynamic scale, second without:

![image_grid](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/62292e57-ac11-4e9b-99e8-02084e95dc17)

### Lost workflows for these as they were done during testing but proves the idea of making SD better at different resolutions:

The temperature was applied to all layers except input 1 and 2, output 9, 10 and 11. At 0.71. Only on self-attention. Using SD v1-5-pruned-emaonly. Resolution at 1024*512. The temperature matches 1/2**2 and the surface is double the normal resolution.

![combined_pair_2](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/5e5403ea-2cb3-462c-a9f1-6cc7b1ddbaea)
![combined_pair_1](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/84fed1e4-a7ba-4f2a-8562-e3573f0aab8f)
![combined_pair_3](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/c6703c21-0d63-404e-9bf8-3a7c580f59e7)

Workflow:

![1](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/6b1bd8b1-c11f-4fde-9ecf-4ea546acad0e)


## Troubleshooting / Q and A

- KeyError: 'input_1': You mixed up the SDXL/SD1 nodes
- Something the 'scale' argument: You need to update pytorch
- Windows error message when starting ComfyUI: You updated pytorch but not xformers (that update goes fast)
- Images are more burned than my pizza when I code: temperature too low
- Images are blurry like I removed my glasses: temperature too high
- Does it work for such model: Probably! There is a node for all models. The model-specific only benefit from an automatic scale if you set them at zero.
- It is compatible with such extension: I can only say that if you want to use the Automatic-CFG with it, you should plug these nodes BEFORE it. For the other patches I do not know. Some may replace the attention function if they do patch the transformer layers.
- Which are the best values? This is uncharted territory. The CLIP temperature is more forgetful. The UNET a lot less, yet something near 0.71 seems overall
- Out of memory error: It uses pytorch attention. It comes from you ;)



# Patreon

Give an incentive to contributors:

https://www.patreon.com/extraltodeus

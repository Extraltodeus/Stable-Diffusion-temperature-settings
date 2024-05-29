# Stable-Diffusion-temperature-settings
Provides the ability to set the temperature for both UNET and CLIP. For ComfyUI.

## The nodes

![image](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/2d04cb28-2a1d-4384-8e62-9e9a6b0829dc)

## Specifities

- The CLIP patch ignores the connections and patches the model within the memory. Simply disconnecting it does not revert the behavior. To revert to default behavior set it at 1 or reload the model without the node connected. It is the only node ignoring the connections and does not modify anything but the connected CLIP model.
- For SD1 and SDXL nodes: settings the temperature at zero will use a dynamic scale proportional to the resolution. It is good for lower resolutions but not on point for higher.

## Usage

Like any other model patch:

![image](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/32b73433-df6a-4c49-99a6-5ddf21a4777a)

## Requirements:

Requires pytorch 2.3 and above.

## Interesting side-effect:

Changing the UNET temperature allows to obtain better results a different resolutions. While it is not the full solution to the scaling issues of Stable Diffusion, it is a strong clue indicating the possibility to sample at much higher (or lower) resolutions.

### Here using SDXL at a resolution of 328x328. First row temperature at 1, second row using dynamic scaled attention:

![image_grid](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/7b5b2ffb-f621-4eca-9f97-04f78c2eaf7c)

### Non cherry-picked SD v1-5-pruned-emaonly at 512*1024, first row with the dynamic scale, second without:

![image_grid](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/62292e57-ac11-4e9b-99e8-02084e95dc17)

### Lost workflows for these as they were done during testing but proves the idea of making SD better at different resolutions:

The temperature was applied to all layers except input 1 and 2, output 9, 10 and 11. At 0.71. Only on self-attention.

![combined_pair_2](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/5e5403ea-2cb3-462c-a9f1-6cc7b1ddbaea)
![combined_pair_1](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/84fed1e4-a7ba-4f2a-8562-e3573f0aab8f)
![combined_pair_3](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/c6703c21-0d63-404e-9bf8-3a7c580f59e7)

### Other simple example with SD 1.5, with/without are obvious enough (these two are workflows):

![01629UI_00001_](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/d6371d3f-ea38-40e4-8215-432214027a78)

![01630UI_00001_](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/967ddf3b-da39-4dca-bd48-02a2f7e28ee0)

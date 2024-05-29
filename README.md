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

### Here using SDXL at a resolution of 328x328. First row is temperature at 1, second and third using the dynamic scaled attention.

![image_grid](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/7b5b2ffb-f621-4eca-9f97-04f78c2eaf7c)

### Non cherry-picked SD v1-5-pruned-emaonly at 512*1024, first row with the dynamic scale, second without:

![image_grid](https://github.com/Extraltodeus/Stable-Diffusion-temperature-settings/assets/15731540/62292e57-ac11-4e9b-99e8-02084e95dc17)


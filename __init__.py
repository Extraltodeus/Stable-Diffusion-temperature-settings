from .nodes import *

NODE_CLASS_MAPPINGS = {
    "Unet Temperature SDXL":UnetTemperaturePatchSDXL,
    "Unet Temperature SD1":UnetTemperaturePatchSD15,
    "Unet Temperature any model":UnetTemperaturePatchAny,
    "Unet Temperature SDXL per layer":UnetTemperaturePatchSDXLpl,
    "Unet Temperature SD1 per layer" :UnetTemperaturePatchSD15pl,
    "CLIP Temperature":CLIPTemperaturePatch,
}

import torch
import math
import types
import comfy.model_management
from functools import partial
from math import *
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP

EPSILON = 1e-16

SD_layer_dims = {
    "SD1" : {"input_1": 4096,"input_2": 4096,"input_4": 1024,"input_5": 1024,"input_7": 256,"input_8": 256,"middle_0": 64,"output_3": 256,"output_4": 256,"output_5": 256,"output_6": 1024,"output_7": 1024,"output_8": 1024,"output_9": 4096,"output_10": 4096,"output_11": 4096},
    "SDXL": {"input_4": 4096,"input_5": 4096,"input_7": 1024,"input_8": 1024,"middle_0": 1024,"output_0": 1024,"output_1": 1024,"output_2": 1024,"output_3": 4096,"output_4": 4096,"output_5": 4096},
    "Disabled":{}
    }

models_by_size = {"1719049928": "SD1", "5134967368":"SDXL"}

def cv_temperature(input_tensor, auto_mode="normal"):
    if "creative" in auto_mode:
        tensor_std  = input_tensor.std()
        temperature = (torch.std(torch.abs(input_tensor - tensor_std))/tensor_std)
        temperature = 1 / temperature
        del tensor_std
    else:
        temperature = torch.std(input_tensor)
    if "squared" in auto_mode:
        temperature = temperature ** 2
    elif "sqrt" in auto_mode:
        temperature = temperature ** .5
    if not "reversed" in auto_mode:
        temperature = 1 / temperature    
    return temperature

def should_scale(mname,lname,q2):
    if mname == "" or mname == "CLIP": return False
    if mname != "Disabled" and lname in SD_layer_dims[mname]:
        if lname not in SD_layer_dims[mname]:
            return False
        return q2 != SD_layer_dims[mname][lname]
    return False

class temperature_patcher():
    def __init__(self, temperature, layer_name = "", model_name="", eval_string="", auto_temp="disabled",
                 Original_scale=512, Target_scale_X=512, Target_scale_Y=512, rescale_adjust=1,
                 scale_before=False,scale_after=False):
        self.temperature = max(temperature,EPSILON)
        self.layer_name  = layer_name
        self.model_name  = model_name
        self.eval_string = eval_string
        self.auto_temp   = auto_temp
        self.Original_scale = Original_scale
        self.Target_scale_X = Target_scale_X
        self.Target_scale_Y = Target_scale_Y
        self.rescale_adjust = rescale_adjust
        self.scale_before = scale_before
        self.scale_after  = scale_after

    def pytorch_attention_with_temperature(self, q, k, v, extra_options, mask=None, attn_precision=None):
        heads = extra_options if isinstance(extra_options, int) else extra_options['n_heads']
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

        if self.auto_temp != "disabled":
            extra_temperature = cv_temperature({'q_': q, 'k_': k, 'v_': v}[self.auto_temp[:2]], self.auto_temp)
        else:
            extra_temperature = 1
        
        temperature_pre_scale = 1
        if self.scale_before:
            if should_scale(self.model_name, self.layer_name,q.size(-2)):
                ldim = SD_layer_dims[self.model_name][self.layer_name]
                if self.eval_string != "":
                    temperature_pre_scale = eval(self.eval_string)
                else:
                    temperature_pre_scale = log(q.size(-2), ldim) # that's the actual resccale, everything else is for testing purpose
            elif (self.Target_scale_X*self.Target_scale_Y) != self.Original_scale**2:
                temperature_pre_scale = log((self.Target_scale_X*self.Target_scale_Y)**.5, self.Original_scale)
        temperature_scale = self.temperature / temperature_pre_scale

        scale = 1 / (math.sqrt(q.size(-1)) * temperature_scale * extra_temperature)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, scale=scale)

        if self.scale_after:
            if should_scale(self.model_name, self.layer_name,q.size(-2)):
                ldim = SD_layer_dims[self.model_name][self.layer_name]
                if self.eval_string != "":
                    out *= eval(self.eval_string)
                else:
                    out *= log(q.size(-2), ldim)
            elif (self.Target_scale_X*self.Target_scale_Y) != self.Original_scale**2:
                out *= log((self.Target_scale_X*self.Target_scale_Y)**.5, self.Original_scale)
        
        if self.rescale_adjust != 1:
            out *= self.rescale_adjust
        
        out = (
            out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        )
        return out

class UnetTemperaturePatch:
    @classmethod
    def INPUT_TYPES(s):
        required_inputs = {}
        required_inputs["model"] = ("MODEL",)
        required_inputs["Temperature"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": 0.01})
        required_inputs["Attention"]   = (["both","self","cross"],)
        required_inputs["Dynamic_Scale_Temperature"] = ("BOOLEAN", {"default": False})
        required_inputs["Dynamic_Scale_Output"] = ("BOOLEAN", {"default": True})
        # required_inputs["eval_string"] = ("STRING", {"multiline": True})
        return {"required": required_inputs}

    TOGGLES = {}
    RETURN_TYPES = ("MODEL","STRING",)
    RETURN_NAMES = ("Model","String",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Temperature"

    def patch(self, model, Temperature, Attention, Dynamic_Scale_Temperature, Dynamic_Scale_Output, Dynamic_Scale_Adjust=1, eval_string="", **kwargs):
        Dynamic_Scale_Attention = Dynamic_Scale_Temperature or Dynamic_Scale_Output
        if not Dynamic_Scale_Attention and Temperature == 1:
            print("Dynamic_Scale_Attention/temperature: no patch applied.")
            return (model, "Fully disabled",)
        if Dynamic_Scale_Attention and str(model.size) in models_by_size:
            model_name = models_by_size[str(model.size)]
            print(f"Model detected for scaling: {model_name}")
        else:
            if Dynamic_Scale_Attention:
                print("No compatible model detected for dynamic scale attention!")
            model_name = "Disabled"

        m = model.clone()
        levels = ["input","middle","output"]
        layer_names = {f"{l}_{n}": True for l in levels for n in range(12)}

        for key, toggle in layer_names.items():
            current_level = key.split("_")[0]
            b_number = int(key.split("_")[1])
            patcher = temperature_patcher(Temperature,layer_name=key,model_name=model_name, eval_string=eval_string,
                                          rescale_adjust=Dynamic_Scale_Adjust, scale_before=Dynamic_Scale_Temperature,
                                          scale_after=Dynamic_Scale_Output)
            if Attention in ["both","self"]:
                m.set_model_attn1_replace(patcher.pytorch_attention_with_temperature, current_level, b_number)
            if Attention in ["both","cross"]:
                m.set_model_attn2_replace(patcher.pytorch_attention_with_temperature, current_level, b_number)

        parameters_as_string = f"Temperature: {Temperature}\nAttention: {Attention}\nDynamic scale: {Dynamic_Scale_Attention}"
        return (m, parameters_as_string,)

class CLIPTemperaturePatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "clip": ("CLIP",),
                              "Temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                            #   "Auto_temp": ("BOOLEAN", {"default": False}) # It's just experimental but uncomment it if you want to try.
                              }}
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/Temperature"
    
    def patch(self, clip, Temperature, Auto_temp=False):
        c = clip.clone()
        def custom_optimized_attention(device, mask=None, small_input=True):
            return temperature_patcher(temperature=Temperature,auto_temp="k_creative" if Auto_temp else "disabled").pytorch_attention_with_temperature
        def new_forward(self, x, mask=None, intermediate_output=None):
            optimized_attention = custom_optimized_attention(x.device, mask=mask is not None, small_input=True)

            if intermediate_output is not None:
                if intermediate_output < 0:
                    intermediate_output = len(self.layers) + intermediate_output

            intermediate = None
            for i, l in enumerate(self.layers):
                x = l(x, mask, optimized_attention)
                if i == intermediate_output:
                    intermediate = x.clone()
            return x, intermediate

        if getattr(c.patcher.model, "clip_g", None) is not None:
            c.patcher.add_object_patch("clip_g.transformer.text_model.encoder.forward", partial(new_forward, c.patcher.model.clip_g.transformer.text_model.encoder))

        if getattr(c.patcher.model, "clip_l", None) is not None:
            c.patcher.add_object_patch("clip_l.transformer.text_model.encoder.forward", partial(new_forward, c.patcher.model.clip_l.transformer.text_model.encoder))

        return (c,)

class CLIPTemperatureWithScalePatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "clip": ("CLIP",),
                              "Temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "Original_scale": ("INT", {"default": 512, "min": 64, "max": 100000.0, "step": 64}),
                              "Target_scale_X": ("INT", {"default": 512, "min": 64, "max": 100000.0, "step": 64}),
                              "Target_scale_Y": ("INT", {"default": 512, "min": 64, "max": 100000.0, "step": 64}),
                              "Dynamic_Scale_Temperature": ("BOOLEAN", {"default": False}),
                              "Dynamic_Scale_Output": ("BOOLEAN", {"default": False})
                              }}
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/Temperature"
    
    def patch(self, clip, Temperature, Dynamic_Scale_Temperature, Dynamic_Scale_Output, Original_scale=512, Target_scale_X=512, Target_scale_Y=512, Scale_Adjust=1):
        c = clip.clone()
        def custom_optimized_attention(device, mask=None, small_input=True):
            return temperature_patcher(temperature=Temperature,Target_scale_X=Target_scale_X,Target_scale_Y=Target_scale_Y,Original_scale=Original_scale, rescale_adjust=Scale_Adjust, scale_before=Dynamic_Scale_Temperature, scale_after=Dynamic_Scale_Output).pytorch_attention_with_temperature
        
        def new_forward(self, x, mask=None, intermediate_output=None):
            optimized_attention = custom_optimized_attention(x.device, mask=mask is not None, small_input=True)

            if intermediate_output is not None:
                if intermediate_output < 0:
                    intermediate_output = len(self.layers) + intermediate_output

            intermediate = None
            for i, l in enumerate(self.layers):
                x = l(x, mask, optimized_attention)
                if i == intermediate_output:
                    intermediate = x.clone()
            return x, intermediate

        if getattr(c.patcher.model, "clip_g", None) is not None:
            c.patcher.add_object_patch("clip_g.transformer.text_model.encoder.forward", partial(new_forward, c.patcher.model.clip_g.transformer.text_model.encoder))

        if getattr(c.patcher.model, "clip_l", None) is not None:
            c.patcher.add_object_patch("clip_l.transformer.text_model.encoder.forward", partial(new_forward, c.patcher.model.clip_l.transformer.text_model.encoder))

        return (c,)

import torch
import math
import types
import comfy.model_management

EPSILON = 1e-16

SD_layer_dims = {
    "SD1" : {"input_1": 4096,"input_2": 4096,"input_4": 1024,"input_5": 1024,"input_7": 256,"input_8": 256,"middle_0": 64,"output_3": 256,"output_4": 256,"output_5": 256,"output_6": 1024,"output_7": 1024,"output_8": 1024,"output_9": 4096,"output_10": 4096,"output_11": 4096},
    "SDXL": {"input_4": 4096,"input_5": 4096,"input_7": 1024,"input_8": 1024,"middle_0": 1024,"output_0": 1024,"output_1": 1024,"output_2": 1024,"output_3": 4096,"output_4": 4096,"output_5": 4096},
    "Disabled":{}
    }

models_by_size = {"1719049928": "SD1", "5134967368":"SDXL"}

def should_scale(mname,lname,q2):
    if mname == "": return False
    if mname != "Disabled" and lname in SD_layer_dims[mname]:
        return q2 != SD_layer_dims[mname][lname]
    return False

class temperature_patcher():
    def __init__(self, temperature, layer_name = "", model_name="", eval_string=""):
        self.temperature = max(temperature,EPSILON)
        self.layer_name  = layer_name
        self.model_name  = model_name
        self.eval_string = eval_string

    def pytorch_attention_with_temperature(self, q, k, v, extra_options, mask=None, attn_precision=None):
        heads = extra_options if isinstance(extra_options, int) else extra_options['n_heads']
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

        scale = 1 / (math.sqrt(q.size(-1)) * self.temperature)
        
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=scale)
        if should_scale(self.model_name, self.layer_name,q.size(-2)):
            if self.eval_string != "":
                out = eval(self.eval_string)
            else:
                out *= math.log(q.size(-2)) / math.log(SD_layer_dims[self.model_name][self.layer_name])

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
        required_inputs["Dynamic_Scale_Attention"] = ("BOOLEAN", {"default": True})
        # required_inputs["eval_string"] = ("STRING", {"multiline": True})
        return {"required": required_inputs}

    TOGGLES = {}
    RETURN_TYPES = ("MODEL","STRING",)
    RETURN_NAMES = ("Model","String",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Temperature"

    def patch(self, model, Temperature, Attention, Dynamic_Scale_Attention, eval_string="", **kwargs):
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
            patcher = temperature_patcher(Temperature,layer_name=key,model_name=model_name, eval_string=eval_string)
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
                              }}
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/Temperature"
    
    def patch(self, clip, Temperature):
        def custom_optimized_attention(device, mask=None, small_input=True):
            return temperature_patcher(Temperature).pytorch_attention_with_temperature
        
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

        clip_encoder_instance = clip.cond_stage_model.clip_l.transformer.text_model.encoder
        clip_encoder_instance.forward = types.MethodType(new_forward, clip_encoder_instance)

        if getattr(clip.cond_stage_model, f"clip_g", None) is not None:
            clip_encoder_instance_g = clip.cond_stage_model.clip_g.transformer.text_model.encoder
            clip_encoder_instance_g.forward = types.MethodType(new_forward, clip_encoder_instance_g)
        
        return (clip,)

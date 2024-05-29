import torch
import math
import types

EPSILON = 1e-4

SD_layer_dims = {
    "SD1" : {"input_1": 4096,"input_2": 4096,"input_4": 1024,"input_5": 1024,"input_7": 256,"input_8": 256,"middle_0": 64,"output_3": 256,"output_4": 256,"output_5": 256,"output_6": 1024,"output_7": 1024,"output_8": 1024,"output_9": 4096,"output_10": 4096,"output_11": 4096},
    "SDXL": {"input_4": 4096,"input_5": 4096,"input_7": 1024,"input_8": 1024,"middle_0": 1024,"output_0": 1024,"output_1": 1024,"output_2": 1024,"output_3": 4096,"output_4": 4096,"output_5": 4096},
    }

layers_SD15 = {
    "input":[1,2,4,5,7,8],
    "middle":[0],
    "output":[3,4,5,6,7,8,9,10,11],
}

layers_SDXL = {
    "input":[4,5,7,8],
    "middle":[0],
    "output":[0,1,2,3,4,5],
}

revert_dim  = lambda x: 8 * math.sqrt(x)
printed_var = ""
def cprint(var):
    global printed_var
    str_var = str(var)
    if printed_var != str_var:
        print(" ",str_var)
        printed_var = str_var

def dynamic_scale_attention(layer_name, model_name, q_size_1, q_size_2, **kwargs):
    return 1 / (math.sqrt(q_size_1) * (SD_layer_dims[model_name][layer_name] ** 0.5 / q_size_2 ** 0.5) ** 0.5)

def temp_non_zero_div(layer_name, model_name, q_size_1, q_size_2, **kwargs):
    return 1 / (math.sqrt(q_size_1) * EPSILON)

auto_temp_methods = {"dsa": dynamic_scale_attention, "clip":temp_non_zero_div}

class temperature_patcher():
    def __init__(self, temperature, layer_name = "", model_name="", eval_string = "", method="medium", base_resolution=(512,512), target_resolution=(512,512)):
        self.temperature = temperature
        self.layer_name  = layer_name
        self.model_name  = model_name
        self.eval_string = eval_string
        self.method      = auto_temp_methods[method]
        self.base_resolution   = base_resolution
        self.target_resolution = target_resolution

    def pytorch_attention_with_temperature(self, q, k, v, extra_options, mask=None, attn_precision=None):
        heads = extra_options if isinstance(extra_options, int) else extra_options['n_heads']
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

        if self.eval_string != "":
            if self.layer_name != "":
                layer_dim = SD_layer_dims[self.model_name][self.layer_name]
            q_size_1 = q.size(-1)
            q_size_2 = q.size(-2)
            c = []
            evals_strings = self.eval_string.split(";")
            if len(evals_strings) > 1:
                for i in range(len(evals_strings[:-1])):
                    c.append(eval(evals_strings[i]))
            scale = eval(evals_strings[-1])
        else:
            scale  = 1 / (math.sqrt(q.size(-1)) * self.temperature) if self.temperature > 0 else \
                self.method(layer_name=self.layer_name, model_name=self.model_name, q_size_1=q.size(-1), q_size_2=q.size(-2),
                            base_resolution=self.base_resolution,target_resolution=self.target_resolution)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False,scale=scale)
        out = (
            out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        )
        return out

class UnetTemperaturePatch:
    @classmethod
    def INPUT_TYPES(s):
        if not s.ANY_MODEL:
            required_inputs = {f"{key}_{layer}": ("BOOLEAN", {"default": True}) for key, layers in s.TOGGLES.items() for layer in layers}
        else:
            required_inputs = {}
        required_inputs["model"] = ("MODEL",)
        required_inputs["Temperature"]  = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": 0.01})
        required_inputs["Attention"]    = (["both","self","cross"],)
        # required_inputs["Method"]       = (["strong","medium","light"],)
        # required_inputs["eval_string"] = ("STRING", {"multiline": True})
        return {"required": required_inputs}
    
    ANY_MODEL  = False
    LAYER_NAME = None
    TOGGLES = {}
    RETURN_TYPES = ("MODEL","STRING",)
    RETURN_NAMES = ("Model","String",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Temperature"

    def patch(self, model, Temperature, Attention, Method="dsa", eval_string="", **kwargs):
        model_name = self.__class__.MODEL_NAME
        any_model  = self.__class__.ANY_MODEL

        if not any_model:
            layer_names = kwargs
        else:
            layer_names = {f"{l}_{n}": True for l in ["input", "middle", "output"] for n in range(12)}

        m = model.clone()
        levels = ["input","middle","output"]
        parameters_output = {level:[] for level in levels}
        
        for key, toggle_enabled in layer_names.items():
            current_level = key.split("_")[0]
            if current_level in levels and toggle_enabled:
                b_number = int(key.split("_")[1])
                parameters_output[current_level].append(b_number)
                patcher = temperature_patcher(Temperature,method=Method if model_name in ["SDXL","SD1"] else "clip",layer_name=key,model_name=model_name,eval_string=eval_string)

                if Attention in ["both","self"]:
                    m.set_model_attn1_replace(patcher.pytorch_attention_with_temperature, current_level, b_number)
                if Attention in ["both","cross"]:
                    m.set_model_attn2_replace(patcher.pytorch_attention_with_temperature, current_level, b_number)

        parameters_as_string = "\n".join(f"{k}: {','.join(map(str, v))}" for k, v in parameters_output.items())
        parameters_as_string = f"Temperature: {Temperature}\n{parameters_as_string}\nAttention: {Attention}"
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
        print(f"\n\n\nThe CLIP patch ignores the connection. Set at 1 to get default behavior. Or reload the model without this node.\n\n\n")
        def custom_optimized_attention(device, mask=None, small_input=True):
            return temperature_patcher(Temperature,method="clip").pytorch_attention_with_temperature
        
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

UnetTemperaturePatchSDXL = type("Unet Temperature SDXL", (UnetTemperaturePatch,), {"TOGGLES": layers_SDXL,"MODEL_NAME":"SDXL","ANY_MODEL": True})
UnetTemperaturePatchSD15 = type("Unet Temperature SD1",  (UnetTemperaturePatch,), {"TOGGLES": layers_SD15,"MODEL_NAME":"SD1", "ANY_MODEL": True,})
UnetTemperaturePatchSDXLpl = type("Unet Temperature SDXL per layer", (UnetTemperaturePatch,), {"TOGGLES": layers_SDXL,"MODEL_NAME":"SDXL","ANY_MODEL": False})
UnetTemperaturePatchSD15pl = type("Unet Temperature SD1 per layer",  (UnetTemperaturePatch,), {"TOGGLES": layers_SD15,"MODEL_NAME":"SD1", "ANY_MODEL": False})
UnetTemperaturePatchAny  = type("Unet Temperature any model", (UnetTemperaturePatch,), {"MODEL_NAME":"SD1","ANY_MODEL": True})
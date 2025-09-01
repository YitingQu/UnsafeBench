import base64
import requests
import json
from pathlib import Path
import os, sys
import pandas as pd
import torch
from PIL import Image
import requests
from io import BytesIO

API_KEY = os.getenv("OPENAI_API_KEY", None)

VLM2ModelPaths = {"gpt-4v": "gpt-4-vision-preview",
                  "llava-v1.5-7b": "liuhaotian/llava-v1.5-7b",
                  "instructblip-7b": ""
                  }

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_path):
    if image_path.startswith("http") or image_path.startswith("https"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return image

def load_images(image_paths):
    out = []
    for image_path in image_paths:
        image = load_image(image_path)
        out.append(image)
    return out

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_type(image_path):
    """ Returns the type of the image (e.g., JPEG, PNG) """
    with Image.open(image_path) as img:
        return img.format.lower()
    
def load_vlm(model_name):
    
    if "gpt" in model_name:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY)
        return client
    
    elif "llava" in model_name:
        from llava.model import LlavaLlamaForCausalLM
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model =  LlavaLlamaForCausalLM.from_pretrained(VLM2ModelPaths[model_name], 
                                                    torch_dtype=torch.float16).cuda().eval()

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        
        vision_tower.to(torch.float16).cuda()
        return model
    
    elif "instructblip" in model_name:
        from lavis.models import load_model_and_preprocess
        model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device="cuda")
        model.eval()
        return model
    
def inference(model_name, model, image_path, prompt, **gen_kwargs):
    
    if "gpt" in model_name:
        
        gen_kwargs = {
        "temperature": gen_kwargs["temperature"],
        "max_tokens": gen_kwargs["max_new_tokens"],
        "top_p": gen_kwargs["top_p"],
        }
        
        base64_image = encode_image(image_path)

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, 
                                                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]
                    }
                    ]
        completion = model.chat.completions.create(
            model=VLM2ModelPaths[model_name],
            store=True,
            messages=messages,
            **gen_kwargs
        )
        # print(completion)
        output = completion.choices[0].message.content
        return output
    
    elif "llava" in model_name:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IMAGE_PATCH_TOKEN
        )
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.utils import disable_torch_init
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            get_model_name_from_path,
            KeywordsStoppingCriteria,
        )
        
        disable_torch_init()
        
        # config tokenizer
        tokenizer = AutoTokenizer.from_pretrained(VLM2ModelPaths[model_name], use_fast=False)
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        
        model.resize_token_embeddings(len(tokenizer))
        
        # load image processor
        image_processor = model.get_vision_tower().image_processor
        
        # generate
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if model.config.mm_use_im_start_end:
            prompt = image_token_se + "\n" + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv_mode = "llava_v1"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        images = [load_image(image_path)]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        model.config.use_cache = True
        model.config.cache_shape = (
            input_ids.shape[-1] + gen_kwargs["max_new_tokens"] + model.get_vision_tower().num_patches,
        )
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                **gen_kwargs
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        response = outputs.strip()
        return response
    
    elif "instructblip" in model_name:
        from lavis.models import load_preprocess_only
        
        gen_kwargs = {
        "temperature": gen_kwargs["temperature"],
        "max_length": gen_kwargs["max_new_tokens"],
        "top_p": gen_kwargs["top_p"],
        }
        
        vis_processors, _ = load_preprocess_only(name="blip2_vicuna_instruct", model_type="vicuna7b")
        raw_image = Image.open(image_path).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).cuda()

        response = model.generate({"image": image, "prompt": prompt}, **gen_kwargs)
        response = [res.strip("<s>").strip() for res in response]
        return response
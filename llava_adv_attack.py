import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from torchvision.utils import save_image
from llava.model.builder import load_pretrained_model
from PIL import Image
from llava_utils import prompt_wrapper, visual_attacker
import sys, json
from pathlib import Path
from unsafe_datasets import *
from llava.constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN
)
from utils import load_LLM_output_converter, PromptTemplate
import tqdm

def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--attack_types", nargs="+", default=["GN", "FGSM", "PGD", "DeepFool"])
    parser.add_argument("--seed", type=int, default=2023, help="seed of sampling attack images")
    parser.add_argument('--eps', type=float, default=0.01, help="epsilon of the attack budget")
    parser.add_argument("--K", type=int, default=500, help="the number of samples to attack")
    parser.add_argument("--batch_size", type=int, default=1, choices=[1], help="currently only support batch size 1")
    parser.add_argument("--prediction_path", type=str, default="./outputs/effectiveness/UnsafeBench", help="the path to the prediction results of llava on UnsafeBench")
    parser.add_argument("--save_dir", type=str, default='./outputs/robustness',
                        help="save directory")

    args = parser.parse_args()
    return args

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

args = parse_args()

def load_pretrained_model(lora_path, model_base, device):
    compute_dtype = torch.float16
    
    from llava.model import LlavaLlamaForCausalLM
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    model =  LlavaLlamaForCausalLM.from_pretrained(model_base,
                                                   device_map={"": device}, 
                                                   torch_dtype=compute_dtype)
    
    model.config.torch_dtype = compute_dtype
    
    if lora_path is not None:
        model = PeftModel.from_pretrained(
                model,
                lora_path,
                is_trainable=False,
            )
        model = model.merge_and_unload()

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device).to(compute_dtype)
    vision_tower.requires_grad_(False)
    image_processor = vision_tower.image_processor
    
    mm_projector = model.get_model().mm_projector
    mm_projector.to(device=device).to(compute_dtype)
    mm_projector.requires_grad_(False)
    
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    return tokenizer, model, image_processor, context_len

tokenizer, model, image_processor, context_len = load_pretrained_model(
    args.model_path, args.model_base, device="cuda"
)
model.eval()

# load roberta output converter
roberta_tokenizer, roberta = load_LLM_output_converter()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

result_df = pd.DataFrame()

for source in SOURCES:
    # load attack dataset
    attack_dataset = random_draw_testing_adv_samples(classifier_name="llava-v1.5-7b", 
                                                    prediction_path=args.prediction_path,
                                                    source=source, 
                                                    K=args.K,
                                                    seed=args.seed)

    dataloader = DataLoader(attack_dataset, batch_size=args.batch_size, shuffle=False)

    my_attacker = visual_attacker.Attacker(args, model, tokenizer,roberta, roberta_tokenizer, device=model.device)

    prompt_templates = PromptTemplate()
    user_prompt = prompt_templates.templates[0] # use the base prompt to conduct the attack
    print("user prompt is ", user_prompt)

    text_prompt_template = prompt_wrapper.prepare_text_prompt(user_prompt=user_prompt)

    target_mapping = {
        0: "unsafe", # originally safe -> unsafe
        1: "safe" # originally unsafe -> safe
    }

    for attack_type in args.attack_types:
        is_adv_list = []

        for idx, batch in enumerate(tqdm.tqdm(dataloader)):

            image_filenames, labels = batch
            targets = [target_mapping[int(label)] for label in labels]
            
            image = [load_image(img) for img in image_filenames]
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()
            image = image.to(torch.float16)
            
            if attack_type == "PGD":
                x_adv, is_adv = my_attacker.attack_PGD(text_prompt_template,
                                                                        img=image, 
                                                                        targets=targets,
                                                                        num_iter=100, alpha=0.01/0.3,
                                                                        epsilon=args.eps)
            elif attack_type == "FGSM":
                x_adv, is_adv = my_attacker.attack_FGSM(text_prompt_template,
                                                                    img=image, 
                                                                    targets=targets,
                                                                    epsilon=args.eps) # 1 step attack
            elif attack_type == "GN":
                x_adv, is_adv = my_attacker.attack_GN(text_prompt_template,
                                                                    img=image, 
                                                                    targets=targets,
                                                                    epsilon=args.eps) # 1 step attack
            if attack_type == "DeepFool":
                x_adv, is_adv = my_attacker.attack_DeepFool(text_prompt_template,
                                                                        img=image, 
                                                                        targets=targets,
                                                                        num_iter=100,
                                                                        epsilon=args.eps)

            is_adv_list.extend([bool(val) for val in is_adv])
            
            # print(is_adv)

        result = {"is_adv": is_adv_list}
        json.dump(result, open(f"{args.save_dir}/llava_{source}_{attack_type}_{args.seed}.json", "w"))
        print(f"Results saved to {args.save_dir}/llava_{source}_{attack_type}_{args.seed}.json")
        
        RA = 1 - np.mean(is_adv_list)
        result_df.loc[source, attack_type] = RA
        print(source, attack_type, "Robust Accuracy:", np.round(RA, 3))

print("Summary of Robust Accuracy for LLaVA:")
print(result_df)
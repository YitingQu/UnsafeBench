import os, sys, json
import torch
from PIL import Image
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch.utils

current_file_path = Path(__file__).resolve()
parent_directory = current_file_path.parent.parent
sys.path.append(str(parent_directory))
from unsafe_datasets import *
from utils import *

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def design_prompt_template(idx, augment_mode, prompt_template, image_label, image_category, category2instruction):

    setup_seed(idx)
        
    all_categories = list(category2instruction.keys())
    held_out_categories = all_categories.copy()
    held_out_categories.remove(image_category)
    random.shuffle(held_out_categories)
    
    k_min = 1
    k_max = max(1, len(held_out_categories))
    sample_num = random.randint(k_min, k_max)  # inclusive

    sample_categories = held_out_categories[:sample_num]
    
    if augment_mode == "original":
        # add the image category itself
        sample_categories.append(image_category)
        random.shuffle(sample_categories)
        sample_instructions = [category2instruction[category] for category in sample_categories]
        
        sample_instructions_text = "\n".join(sample_instructions)
        
        input_prompt = prompt_template.replace("[instruction]", sample_instructions_text)
        
        if image_label == "Safe":
            output_prompt = "Safety Status: Safe, Unsafe Category: None"
        elif image_label == "Unsafe":
            output_prompt = f"Safety Status: Unsafe, Unsafe Category: {image_category}"
        return input_prompt, output_prompt
    
    elif augment_mode == "negative":
        random.shuffle(sample_categories)
        sample_instructions = sample_categories
        # sample_instructions = [category2instruction[category] for category in sample_categories]
        
        sample_instructions_text = "\n".join(sample_instructions)
        
        input_prompt = prompt_template.replace("[instruction]", sample_instructions_text)
        # alwasy output safe in this case
        output_prompt = "Safety Status: Safe, Unsafe Category: None"
        return input_prompt, output_prompt
        
def prepare_instructions_for_llava(mode):

    base_dir = os.path.dirname(os.path.dirname(__file__))
    image_root = os.path.join(base_dir, "data", "UnsafeBench")

    label_mappings = {0: "Safe", 1: "Unsafe"}
    categories, instructions = list_unsafe_instruction()
    category2instruction = dict(zip(categories, instructions))
            
    sources = ["Laion5B", "Lexica"]
    prompt_template = open("prompts/prompt_template.txt", "r").read()
    
    data_comprehensive = []
    unsafe_data = []
    
    idx = 0
    for category, instruction in zip(categories, instructions):
        
        for source in sources:
            dataset = UnsafeBenchDataset(image_root=image_root, source=source, category=category, partition="train")
            for _, item in enumerate(dataset):
                image_fname, label = item
                image_fname = image_fname.replace(base_dir+"/", "")
                label = label_mappings[label]
                
                if label == "Unsafe":
                    unsafe_data.append([image_fname, label, category])
                
                # perturb order and number of categories      
                prompt, gpt_output = design_prompt_template(idx, "original", prompt_template, label, category, category2instruction)
                
                final_prompt = [{"from": "human", "value": f"<image>\n{prompt}"}, {"from":"gpt", "value": gpt_output}]
                data_comprehensive.append(
                    {
                        "id": str(idx),
                        "image": image_fname,
                        "conversations": final_prompt
                    }
                )
                
                # data augmentation: remove certain category and change the label of originally unsafe image to Safe
                if mode == "negative_sampling":
                    if label == "Unsafe":
                        prompt, gpt_output = design_prompt_template(idx, "negative", prompt_template, label, category, category2instruction)
                        final_prompt = [{"from": "human", "value": f"<image>\n{prompt}"}, {"from":"gpt", "value": gpt_output}]
                        data_comprehensive.append(
                            {
                                "id": f"{idx}_augment",
                                "image": image_fname,
                                "conversations": final_prompt
                            }
                        )
                
                idx += 1
    
    # balance the dataset
    def get_data_num(data, safe=True):
        if safe:
            return len([item for item in data if item["conversations"][1]["value"].startswith("Safety Status: Safe")])
        else:
            return len([item for item in data if item["conversations"][1]["value"].startswith("Safety Status: Unsafe")])
    
    safe_num = get_data_num(data_comprehensive, safe=True)
    unsafe_num = get_data_num(data_comprehensive, safe=False)
    
    print(f"there are {safe_num} safe data and {unsafe_num} unsafe data")
    
    add_unsafe_num = max((safe_num - unsafe_num), 0)
    add_unsafe_data = random.choices(unsafe_data, k=add_unsafe_num)
    print(f"add {add_unsafe_num} data to balance the dataset")
    
    for (image_fname, label, category) in add_unsafe_data:
        prompt, gpt_output = design_prompt_template(idx, "original", prompt_template, label, category, category2instruction)
        
        final_prompt = [{"from": "human", "value": f"<image>\n{prompt}"}, {"from":"gpt", "value": gpt_output}]
        data_comprehensive.append(
            {
                "id": str(idx),
                "image": image_fname,
                "conversations": final_prompt
            }
        )
        
        idx += 1
        
    # lastly, shuffle it
    random.shuffle(data_comprehensive)
    
    safe_num = get_data_num(data_comprehensive, safe=True)
    unsafe_num = get_data_num(data_comprehensive, safe=False)
    print(f"safe: {safe_num}, unsafe: {unsafe_num}, total: {len(data_comprehensive)}")
    
    return data_comprehensive

if __name__ == "__main__":
    
    mode = "negative_sampling"
    data = prepare_instructions_for_llava(mode)

    json.dump(data, open(f"prompts/training_prompts.json", "w"), indent=2)

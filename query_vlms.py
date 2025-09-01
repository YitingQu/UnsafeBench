import torch
from PIL import Image
import argparse
import json
from pathlib import Path
import sys, os
from unsafe_datasets import fetch_merged_UnsafeBench_dataset, list_unsafe_instruction, SOURCES
from utils import PromptTemplate, load_LLM_output_converter
from vlms import load_vlm, inference
from tqdm import tqdm

def main(args):

    # load vlm
    model = load_vlm(args.model_name)
    
    # load prompts
    CATEGORIES, INSTRUCTIONS = list_unsafe_instruction()
    prompt_templates = PromptTemplate()
    
    for prompt_index in range(len(prompt_templates.templates)):
        print(f"Using prompt: {prompt_templates.templates[prompt_index]}")
        for source in SOURCES:
    
            for category, instruction in zip(CATEGORIES, INSTRUCTIONS):
                
                result = []
                dataset = fetch_merged_UnsafeBench_dataset(source=source, category=category)
                
                prompts = prompt_templates.get_prompt(instruction)
                final_prompt = prompts[prompt_index]
                
                gen_kwargs = {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "num_beams": args.num_beams,
                    "max_new_tokens": args.max_new_tokens,
                }
                
                for idx, item in enumerate(tqdm(dataset, desc=f"Querying for {source}:{category}")):
                    image_fname = item[0]
                    image_path = os.path.join(args.image_root, image_fname)
                    
                    response = inference(args.model_name, model, image_path, final_prompt, **gen_kwargs)
                    # print(f"Response for item {idx}: {response}")

                    result.append({
                        "image_fname": image_fname,
                        "response": response
                    })
                    
                # save results
                save_dir = os.path.join(args.save_dir, args.model_name, "prompt_"+str(prompt_index))
                Path(save_dir).mkdir(exist_ok=True, parents=True)
                json.dump(result, open(os.path.join(save_dir, f"{source}_{category}.json"), "w"))
                print(f"Results saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # customized
    parser.add_argument("--model_name", type=str, default="llava-v1.5-7b", choices=["llava-v1.5-7b", "instructblip-7b", "gpt-4v"])
    parser.add_argument("--image_root", type=str, default="", help="Root directory for images")
    parser.add_argument("--save_dir", type=str, default="outputs/vlm_responses")
    
    # generation settings
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    main(args)
import argparse
import os
import random
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
from lavis.models import load_model_and_preprocess
from blip_utils import visual_attacker
from torch.utils.data import DataLoader
from torch.nn import functional as F
import json
from unsafe_datasets import *
from utils import load_LLM_output_converter, PromptTemplate
import tqdm

def save_adv_image(image_tensor, save_dir=None, denormalize=True):
    image_tensor = image_tensor.detach().cpu()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    # Denormalize the image
    if denormalize:
        image_tensor = image_tensor * std + mean

    image = to_pil_image(image_tensor.squeeze(0))
    image.save(save_dir)

def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
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

args = parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model, vis_processor, _ = load_model_and_preprocess(
        name='blip2_vicuna_instruct',
        model_type='vicuna7b',
        is_eval=True,
        device=device,
    )
model.eval()

# load roberta output converter
roberta_tokenizer, roberta = load_LLM_output_converter()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

result_df = pd.DataFrame()

for source in SOURCES:
        
    # load attack dataset
    attack_dataset = random_draw_testing_adv_samples(classifier_name="instructblip-7b", 
                                                    prediction_path=args.prediction_path,
                                                    source=source, 
                                                    K=args.K, 
                                                    seed=args.seed)
    dataloader = DataLoader(attack_dataset, batch_size=args.batch_size, shuffle=False)

    my_attacker = visual_attacker.Attacker(args, model, roberta, roberta_tokenizer, device=model.device)

    prompt_templates = PromptTemplate()
    text_prompt_template = prompt_templates.templates[0] # use the base prompt to conduct the attack
    print("user prompt is ", text_prompt_template)

    target_mapping = {
        0: "unsafe", # originally safe -> unsafe
        1: "safe" # originally unsafe -> safe
    }

    for attack_type in args.attack_types:
        
        is_adv_list = []

        for idx, batch in enumerate(tqdm.tqdm(dataloader)):

            image_filenames, labels = batch
            targets = [target_mapping[int(label)] for label in labels]

            image = [Image.open(img).convert('RGB') for img in image_filenames]
            image =  vis_processor["eval"](image[0]).unsqueeze(0).to(device)
            
            if attack_type == "PGD":
                adv_images, is_adv = my_attacker.attack_PGD(text_prompt_template,
                                                                        img=image, 
                                                                        targets=targets,
                                                                        num_iter=100, alpha=0.01/0.3,
                                                                        epsilon=args.eps)
            elif attack_type == "FGSM":
                adv_images, is_adv = my_attacker.attack_FGSM(text_prompt_template,
                                                                    img=image, 
                                                                    targets=targets,
                                                                    epsilon=args.eps)
            elif attack_type == "GN":
                adv_images, is_adv = my_attacker.attack_GN(text_prompt_template,
                                                                    img=image, 
                                                                    targets=targets,
                                                                    epsilon=args.eps)
                
            elif attack_type == "DeepFool":
                adv_images, is_adv = my_attacker.attack_DeepFool(text_prompt_template,
                                                                        img=image,
                                                                        targets=targets,
                                                                        num_iter=100,
                                                                        epsilon=args.eps)

            is_adv_list.extend([bool(val) for val in is_adv])

        result = {"is_adv": is_adv_list}
        json.dump(result, open(f"{args.save_dir}/instructblip_{source}_{attack_type}_{args.seed}.json", "w"))
        print(f"Results saved to {args.save_dir}/instructblip_{source}_{attack_type}_{args.seed}.json")
        
        RA = 1 - np.mean(is_adv_list)
        result_df.loc[source, attack_type] = np.round(RA, 3)
        print(source, attack_type, "Robust Accuracy:", RA)

print("Summary of Robust Accuracy for InstructBLIP:")
print(result_df)

# save results
out_path = os.path.join(args.save_dir, f"robustness_InstructBLIP.xlsx")
result_df.to_excel(out_path, index=False)
print(f"results saved to {out_path}")
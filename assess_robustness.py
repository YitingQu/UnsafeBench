import os, sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
import open_clip
import pickle

import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import tqdm
import fire
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
import foolbox as fb
from classifiers import load_conventional_classifier
from unsafe_datasets import *
import tensorflow as tf

device = "cuda" if torch.cuda.is_available() else "cpu"

def init_attack(attack_name):
    if attack_name == "FGSM":
        attack = fb.attacks.LinfFastGradientAttack()
    elif attack_name == "PGD":
        attack = fb.attacks.LinfProjectedGradientDescentAttack(steps=100)
    elif attack_name == "DeepFool":
        attack = fb.attacks.LinfDeepFoolAttack(steps=100, candidates=2)
    elif attack_name == "GN":
        attack = fb.attacks.L2AdditiveGaussianNoiseAttack()
    return attack
   
def launch_attack(classifier_name,
                  source, 
                  attack_name, 
                  prediction_path,
                  batch_size, 
                  seed, 
                  epsilon,
                  K):
    
    classifier = load_conventional_classifier(classifier_name, device)
    classifier.eval()
    
    if classifier_name != "NudeNet":
        fmodel = fb.PyTorchModel(classifier, bounds=(-3, 3), preprocessing=None)
    elif classifier_name == "NudeNet":
        fmodel = fb.TensorFlowModel(classifier.nsfw_model, bounds=(-1, 1), preprocessing=None)
    
    # init attack
    attack = init_attack(attack_name)
    
    # load samples to attack
    attack_dataset = random_draw_testing_adv_samples(classifier_name=classifier_name, 
                                                     prediction_path=prediction_path,
                                                     source=source, 
                                                     K=K, 
                                                     seed=seed)
    dataloader = DataLoader(attack_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    is_adv_list = []
    for batch in tqdm.tqdm(dataloader):
        image_fnames, labels = batch
        labels = labels.to(device)
        image_tensor = classifier.preprocess_images(image_fnames)
        
        if classifier_name == "NudeNet":
            labels = torch.ones_like(labels) - labels # we flip the label here as NudeNet considers 0 as unsafe
            labels = tf.convert_to_tensor(labels.cpu(), dtype=tf.int32)
            
            _, clipped, is_adv = attack(fmodel, image_tensor, labels, epsilons=epsilon)
            image_tensor = torch.from_numpy(image_tensor.numpy())
            clipped = torch.from_numpy(clipped.numpy())
        else:
            image_tensor = image_tensor.to(device)
            _, clipped, is_adv = attack(fmodel, image_tensor, labels, epsilons=epsilon)
            is_adv = is_adv.detach()
            
            new_predictions = classifier.classify(clipped)
            new_predictions = new_predictions.argmax(dim=1).detach().cpu()
            
            is_adv = (new_predictions != labels.detach().cpu())
            
        is_adv = is_adv.cpu().numpy()
        
        is_adv_list.extend(is_adv.tolist())
        
    return is_adv_list

def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame()
    
    for classifier_name in args.classifiers:

        for source in SOURCES:
            for attack_name in args.attack_types:
                
                is_adv_list = launch_attack(classifier_name=classifier_name, 
                                            source=source,
                                            attack_name=attack_name, 
                                            prediction_path=args.prediction_path,
                                            batch_size=args.batch_size, 
                                            seed=args.seed, 
                                            epsilon=args.eps,
                                            K=args.K)
                
                result = {"is_adv_list": is_adv_list}
                # calculate robust accuracy
                RA = 1 - np.mean(is_adv_list)
                result_df.loc[source, classifier_name] = RA
                print(source, classifier_name, attack_name, "Robust Accuracy:", np.round(RA, 3))

                with open(os.path.join(args.save_dir, f"{classifier_name}_{source}_{attack_name}_{args.seed}.json"), "w") as f:
                    json.dump(result, f)
                print(f"{classifier_name}_{source}_{attack_name} is done")
    
    print("The summary of robust accuracy:")
    print(result_df)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifiers", nargs="+", default=["Q16", "MultiHeaded", "SD_Filter", "NSFW_Detector", "NudeNet"])
    parser.add_argument("--attack_types", nargs="+", default=["GN", "FGSM", "PGD", "DeepFool"])
    parser.add_argument("--eps", type=float, default=0.01, help="the maximum perturbation for each pixel")
    parser.add_argument("--K", type=int, default=500, help="the number of samples to attack")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2023, choices=[2023, 2024, 2025])
    parser.add_argument("--prediction_path", type=str, default="./outputs/effectiveness/UnsafeBench", help="the path to the prediction results of classifiers on UnsafeBench")
    parser.add_argument("--save_dir", type=str, default="./outputs/robustness")
    args = parser.parse_args()
    
    main(args)
    
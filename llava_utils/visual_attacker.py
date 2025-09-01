import torch
from tqdm import tqdm
import random
import sys
# Set up your project with proper __init__.py files and import normally
from llava_utils import generator
from llava_utils import prompt_wrapper
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor


def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


class Attacker:

    def __init__(self, args, model, tokenizer, roberta, roberta_tokenizer, device='cuda:0', is_rtp=False, bounds=(-3, 3)):

        self.args = args
        self.model = model
        self.tokenizer= tokenizer
        self.roberta = roberta.to(device)
        self.roberta_tokenizer = roberta_tokenizer
        self.device = device
        self.is_rtp = is_rtp
        self.bounds = bounds
        
        self.loss_buffer = []

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.get_model().get_vision_tower().requires_grad_(False)
        
        self.roberta.eval()
        self.roberta.requires_grad_(False)
    
    @torch.no_grad()
    def classify_response(self, response):
        _out = self.roberta_tokenizer(response, return_tensors="pt", max_length=512, truncation=True)
        input_ids = _out.input_ids.to(self.device)
        attention_mask = _out.attention_mask.to(self.device)
        logits = self.roberta(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)
        return preds
    
    def attack_GN(self, text_prompt, img, targets, epsilon=0.01):
        
        batch_size = len(targets)
        assert batch_size == len(targets) == img.shape[0]
        
        my_generator = generator.Generator(model=self.model, tokenizer=self.tokenizer)

        x = img.clone().cuda()
        x.requires_grad = True  # Ensure gradients can be calculated for x

        prompt = prompt_wrapper.Prompt(self.model, self.tokenizer, text_prompts=text_prompt, device=self.device)
        batch_targets = targets

        # Generate Gaussian noise with zero mean and unit variance
        adv_noise = torch.randn_like(x).cuda()
        
        # Calculate the L2 norm of the noise
        adv_noise_norm = torch.norm(adv_noise.view(adv_noise.size(0), -1), dim=-1, keepdim=True).view(-1, 1, 1, 1)
        adv_noise = adv_noise / adv_noise_norm * epsilon
        adv_noise.data = adv_noise.data.clamp(-epsilon, epsilon)
        
        x_adv = x + adv_noise
        x_adv = x_adv.clamp(self.bounds[0], self.bounds[1])  # Ensure x_adv remains a valid image

        # generate response  
        with torch.no_grad():
            response = my_generator.generate(prompt, x_adv)
            response = [response] * batch_size 

        # use roberta to check if the generated response is the same as the target
        preds = self.classify_response(response).detach().cpu().numpy()
        is_adv = []
        for pred, tar in zip(preds, targets):
            tar = 0 if tar == "safe" else 1
            is_adv.append(pred == tar)
         
        # print('current response:', response, " targets:", targets, "is_adv: ", is_adv)
                
        return x_adv.detach().cpu(), is_adv
    
    def attack_FGSM(self, text_prompt, img, targets, epsilon=0.01):
        
        batch_size = len(targets)
        assert batch_size == len(targets) == img.shape[0]
        
        my_generator = generator.Generator(model=self.model, tokenizer=self.tokenizer)

        x = img.clone().cuda()
        x.requires_grad = True  # Ensure gradients can be calculated for x

        prompt = prompt_wrapper.Prompt(self.model, self.tokenizer, text_prompts=text_prompt, device=self.device)
        batch_targets = targets
        
        x_adv = x
        target_loss = self.attack_loss(prompt, x_adv, batch_targets)
        target_loss.backward(retain_graph=True)  # Calculate gradients

        # FGSM update step
        x_adv = x - epsilon * x.grad.sign() # subtracting here as we want to descrease the loss of model generate targets 
        x_adv = x_adv.clamp(self.bounds[0], self.bounds[1])  # Ensure x_adv remains a valid image

        self.loss_buffer.append(target_loss.item())

        # print("target_loss: %f" % (target_loss.item()))

        # generate response  
        with torch.no_grad():
            response = my_generator.generate(prompt, x_adv)
            response = [response] * batch_size 

        # use roberta to check if the generated response is the same as the target
        preds = self.classify_response(response).detach().cpu().numpy()
        is_adv = []
        for pred, tar in zip(preds, targets):
            tar = 0 if tar == "safe" else 1 # why safe here?
            is_adv.append(pred == tar)
         
        # print('current response:', response, " targets:", targets, "is_adv: ", is_adv)
                
        return x_adv.detach().cpu(), is_adv

    def attack_PGD(self, text_prompt, img, targets, num_iter=2000, alpha=1/255, epsilon = 0.01):

        batch_size = len(targets)
        assert batch_size == len(targets) == img.shape[0]
        my_generator = generator.Generator(model=self.model, tokenizer=self.tokenizer)

        adv_noise = torch.rand_like(img, requires_grad=True).cuda() * 2 * epsilon - epsilon

        x = img.clone().cuda()
        adv_noise.data = (adv_noise.data + x.data).clamp(self.bounds[0], self.bounds[1]) - x.data
        adv_noise = adv_noise.cuda().requires_grad_(True)
        adv_noise.retain_grad()

        prompt = prompt_wrapper.Prompt(self.model, self.tokenizer, text_prompts=text_prompt, device=self.device)
        for t in range(num_iter + 1):
            batch_targets = random.sample(targets, batch_size)

            x_adv = x + adv_noise
            # x_adv = normalize(x_adv)
            target_loss = self.attack_loss(prompt, x_adv, batch_targets)
            target_loss.backward(retain_graph=True)
            
            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)
            adv_noise.data = (adv_noise.data + x.data).clamp(self.bounds[0], self.bounds[1]) - x.data
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())

            # print("target_loss: %f" % (target_loss.item()))
            
            if t % 100 == 0:
                x_adv = x + adv_noise
                
                # generate response  
                with torch.no_grad():
                    response = my_generator.generate(prompt, x_adv)
                    # print('>>>', response.lower())
                
                # change this for batch training 
                response = [response] * batch_size

                # use roberta to check if the generated response is the same as the target
                preds = self.classify_response(response).detach().cpu().numpy()
                is_adv = []
                for pred, tar in zip(preds, targets):
                    tar = 0 if tar == "safe" else 1
                    is_adv.append(pred == tar)
                
                # print('current response:', response, " targets:", targets, "is_adv: ", is_adv)
                
                if sum(is_adv) == len(is_adv):
                    # print('adversarial examples found at iteration %d' % t)
                    return x_adv.detach().cpu(), is_adv
        
        return x_adv.detach().cpu(), is_adv

    def attack_DeepFool(self, text_prompt, img, targets, num_iter=2000, overshoot=0.02, epsilon=0.01):
        batch_size = len(img)
        assert batch_size == img.shape[0]
        my_generator = generator.Generator(model=self.model, tokenizer=self.tokenizer)

        # Initialize perturbation as zero
        adv_noise = torch.zeros_like(img, requires_grad=True).cuda()
        x = img.clone().cuda()
        adv_noise.retain_grad()

        prompt = prompt_wrapper.Prompt(self.model, self.tokenizer, text_prompts=text_prompt, device=self.device)

        for t in range(num_iter + 1):
            batch_targets = random.sample(targets, batch_size)

            x_adv = x + adv_noise
            # x_adv_norm = normalize(x_adv)

            target_loss = self.attack_loss(prompt, x_adv.to(self.model.dtype), batch_targets)
            target_loss.backward(retain_graph=True)
            
            grad = adv_noise.grad.detach()
            distance = torch.abs(target_loss) / (torch.norm(grad.view(batch_size, -1), p=1) + 1e-8)
            adv_noise.data = adv_noise.data - grad.sign() * distance.view(-1, 1, 1, 1)
            
            adv_noise.data = (1 + overshoot) * adv_noise.data
            adv_noise.data = (adv_noise.data + x.data).clamp(self.bounds[0], self.bounds[1]) - x.data
   
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())

            # print("target_loss: %f" % (target_loss.item()))
            
            if t % 100 == 0:
                # print('######### Output - Iter = %d ##########' % t)
                adv_noise.data = adv_noise.data.clamp(-epsilon, epsilon)

                x_adv = x + adv_noise
                # x_adv = normalize(x_adv)
                
                # generate response  
                with torch.no_grad():
                    response = my_generator.generate(prompt, x_adv)
                
                # change this for batch training 
                response = [response] * batch_size

                # use roberta to check if the generated response is the same as the target
                preds = self.classify_response(response).detach().cpu().numpy()
                is_adv = []
                for pred, tar in zip(preds, targets):
                    tar = 0 if tar == "safe" else 1
                    is_adv.append(pred == tar)
                
                # print('current response:', response, " targets:", targets, "is_adv: ", is_adv)
                
                if sum(is_adv) == len(is_adv):
                    # print('adversarial examples found at iteration %d' % t)
                    return x_adv.detach().cpu(), is_adv
        
        return x_adv.detach().cpu(), is_adv

    
    def attack_loss(self, prompts, images, targets):
        
        context_length = prompts.context_length
        context_input_ids = prompts.input_ids
        batch_size = len(targets)
        
        if len(context_input_ids) == 1:
            context_length = context_length * batch_size
            context_input_ids = context_input_ids * batch_size

        images = images.repeat(batch_size, 1, 1, 1)

        assert len(context_input_ids) == len(targets), f"Unmathced batch size of prompts and targets {len(context_input_ids)} != {len(targets)}"


        to_regress_tokens = [ torch.as_tensor([item[1:]]).cuda() for item in self.tokenizer(targets).input_ids] # get rid of the default <bos> in targets tokenization.


        seq_tokens_length = []
        labels = []
        input_ids = []

        for i, item in enumerate(to_regress_tokens):

            L = item.shape[1] + context_length[i]
            seq_tokens_length.append(L)

            context_mask = torch.full([1, context_length[i]], -100,
                                      dtype=to_regress_tokens[0].dtype,
                                      device=to_regress_tokens[0].device)
            labels.append( torch.cat( [context_mask, item], dim=1 ) )
            input_ids.append( torch.cat( [context_input_ids[i], item], dim=1 ) )

        # padding token
        pad = torch.full([1, 1], 0,
                         dtype=to_regress_tokens[0].dtype,
                         device=to_regress_tokens[0].device).cuda() # it does not matter ... Anyway will be masked out from attention...


        max_length = max(seq_tokens_length)
        attention_mask = []

        for i in range(batch_size):

            # padding to align the length
            num_to_pad = max_length - seq_tokens_length[i]

            padding_mask = (
                torch.full([1, num_to_pad], -100,
                       dtype=torch.long,
                       device=self.device)
            )
            labels[i] = torch.cat( [labels[i], padding_mask], dim=1 )

            input_ids[i] = torch.cat( [input_ids[i],
                                       pad.repeat(1, num_to_pad)], dim=1 )
            attention_mask.append( torch.LongTensor( [ [1]* (seq_tokens_length[i]) + [0]*num_to_pad ] ) )

        labels = torch.cat( labels, dim=0 ).cuda()
        input_ids = torch.cat( input_ids, dim=0 ).cuda()
        attention_mask = torch.cat(attention_mask, dim=0).cuda()

        with torch.enable_grad():
            outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=labels,
                    images=images,
                )
        loss = outputs.loss

        return loss
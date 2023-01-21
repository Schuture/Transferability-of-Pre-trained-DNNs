import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models


def get_weight_variation(num_classes, state_dict_r, state_dict_A, state_dict_B):
    
    model_r = models.resnet50()
    model_A = models.resnet50()
    model_B = models.resnet50()
        
    model_r.load_state_dict(state_dict_r)
    model_A.load_state_dict(state_dict_A)
    
    num_ftrs = model_B.fc.in_features
    model_r.fc = nn.Linear(num_ftrs, num_classes)  # replace the ImageNet classifier
    model_A.fc = nn.Linear(num_ftrs, num_classes)  # replace the ImageNet classifier
    model_B.fc = nn.Linear(num_ftrs, num_classes)
    
    model_B.load_state_dict(state_dict_B)
    
    weight_variation_r_B = [0] * 50
    weight_variation_A_B = [0] * 50
    
    model_r_param_iter = iter(model_r.named_parameters())
    model_A_param_iter = iter(model_A.named_parameters())
    model_B_param_iter = iter(model_B.named_parameters())
    
    i = 0
    while True:
        name_r, param_r = next(model_r_param_iter)
        name_A, param_A = next(model_A_param_iter)
        name_B, param_B = next(model_B_param_iter)
        if 'conv' in name_r or name_r == 'fc.weight':
            variation_r_B = param_r - param_B
            variation_A_B = param_A - param_B
            weight_variation_r_B[i] = float(variation_r_B.detach().cpu().abs().mean())
            weight_variation_A_B[i] = float(variation_A_B.detach().cpu().abs().mean())
            i += 1
        if name_r == 'fc.weight':
            break
    
    weight_variation_r_B = [weight_variation_r_B[0]] +\
                            [sum(weight_variation_r_B[3*i+1:3*i+4])/3 for i in range(16)] +\
                            [weight_variation_r_B[-1]]
    weight_variation_A_B = [weight_variation_A_B[0]] +\
                            [sum(weight_variation_A_B[3*i+1:3*i+4])/3 for i in range(16)] +\
                            [weight_variation_A_B[-1]]
            
    return weight_variation_r_B, weight_variation_A_B


def vis(weight_variation_r_B, weight_variation_A_B):
    tran_A_B = [weight_variation_r_B[i]/weight_variation_A_B[i] for i in range(len(weight_variation_r_B))]
    print('Layer-wise transferability: ', [round(tran_A_B[i], 2) for i in range(18)])
    tran_overall_A_B = sum(tran_A_B[:-1])/len(tran_A_B[:-1])
    
    plt.figure(figsize=(6,4), dpi=150)
    plt.plot(list(range(1,len(tran_A_B)+1)), tran_A_B, marker='o')
    plt.xlabel('layer')
    plt.ylabel('transferability')
    plt.title('Transferability (mean {:.2f})'.format(tran_overall_A_B))
    plt.show()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate transferability')
    
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('-r', type=str, default='', help='Randomly initialized model')
    parser.add_argument('-a', type=str, default='', help='Model pretrained on task A')
    parser.add_argument('-b', type=str, default='', help='Model finietuned on task B')
    
    args = parser.parse_args()
    
    state_dict_r = torch.load(args.r)
    state_dict_A = torch.load(args.a)
    state_dict_B = torch.load(args.b)
    
    weight_variation_r_B, weight_variation_A_B = get_weight_variation(args.num_classes, state_dict_r, state_dict_A, state_dict_B)
    vis(weight_variation_r_B, weight_variation_A_B)
    











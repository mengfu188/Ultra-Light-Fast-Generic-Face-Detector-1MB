import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model_path')
parser.add_argument('slim_path')

args = parser.parse_args()

state = torch.load(args.model_path)

torch.save(state['model'], args.slim_path)


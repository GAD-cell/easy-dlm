import torch 
import torch.nn as nn
import yaml
import argparse
from muon import MuonClip MuonConfig
from ..utils    import DifusionCollate
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    main(args)


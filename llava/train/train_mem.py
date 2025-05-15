import os
import sys
sys.path.insert(0, os.getcwd())
print(sys.path)
from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

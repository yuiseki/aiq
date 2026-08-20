import os
import fire
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = "1"
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = "1"

from aiq.train import train
from aiq.embed import embed
from aiq.classify import classify
from aiq.label import label
from aiq.extract import extract

def main():
    fire.Fire({
        "label": label,
        "train": train,
        "embed": embed,
        "classify": classify,
        "extract": extract
    })

if __name__ == "__main__":
    main()

import argparse
import datetime
import json
import os

import torch
import yaml

from dataset_covid import get_dataloader
from PropensityNet import load_data
from src.main_model import CoDiS
from src.utils import evaluate, train

import wandb


torch.manual_seed(0)

parser = argparse.ArgumentParser(description="CoDiS COVID runner")
parser.add_argument("--config", type=str, default="covid.yaml")
parser.add_argument("--current_id", type=str, default="covid_causal_14d")
parser.add_argument("--device", default="cuda", help="Device")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.2)
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--train", type=int, default=1)
parser.add_argument("--smoke_test", type=int, default=0, help="1: dataloader/model sanity only")
parser.add_argument(
    "--wandb_mode",
    type=str,
    default="online",
    choices=["disabled", "online", "offline"],
)
args = parser.parse_args()

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_MODE"] = args.wandb_mode

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

data_name = config["dataset"]["data_name"]
if data_name != "covid":
    raise ValueError(f"exe_covid.py expects dataset.data_name='covid', got '{data_name}'")

print(args)
print("Dataset is:")
print(data_name)
print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = f"./save_diffs4/covid/_{args.current_id}_{current_time}/"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

print("Start exe_covid on current_id", args.current_id)
train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
    current_id=args.current_id,
)

propnet = load_data(dataset_name=data_name, current_id=args.current_id)
print("Finish training propnet and fix the parameters")
propnet.eval()

model = CoDiS(config, args.device).to(args.device)

if args.smoke_test:
    batch = next(iter(train_loader))
    with torch.no_grad():
        _ = model(batch, is_train=1, propnet=propnet.to(args.device))
    print("SMOKE TEST PASSED: dataloader + propnet + model forward are OK.")
    raise SystemExit(0)

run = wandb.init(
    project="CoDiS-covid",
    notes="CoDiS-covid",
    name=args.current_id,
)

if args.train:
    wandb.config = {
        "epochs": config["train"]["epochs"],
        "num_steps": config["diffusion"]["num_steps"],
        "lr": config["train"]["lr"],
    }
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        valid_epoch_interval=config["train"]["valid_epoch_interval"],
        foldername=foldername,
        propnet=propnet.to(args.device),
    )
    print("----------------Finish training------------")
    print("---------------Start testing---------------")
    evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)

    directory = "./save_model/covid/" + args.current_id
    os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), directory + "/model_weights.pth")
    run.finish()
else:
    directory = "./save_model/covid/" + args.current_id
    model.load_state_dict(torch.load(directory + "/model_weights.pth"))

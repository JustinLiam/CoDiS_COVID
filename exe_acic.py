import argparse
import datetime
import json
import os
import sys

import torch
import yaml

from src.main_model import CoDiS
from src.utils import train, evaluate
from dataset_acic import get_dataloader

from PropensityNet import load_data
import wandb


torch.manual_seed(0)
parser = argparse.ArgumentParser(description="CoDiS")

parser.add_argument("--config", type=str, default="acic2018.yaml")
parser.add_argument("--current_id", type=str, default="00ea30e866f141d9880d5824a361a76a")


parser.add_argument("--device", default="cuda", help="Device")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.2)
parser.add_argument("--nfold", type=int, default=1, help="for 5-fold test")
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--train", type=int, default=1)
parser.add_argument(
    "--smoke_test",
    type=int,
    default=0,
    help="1: dataloader + propnet + model forward sanity only",
)
parser.add_argument(
    "--wandb_mode",
    type=str,
    default="online",
    choices=["disabled", "online", "offline"],
)
parser.add_argument('--gpu', type=int, default=0, help='gpu ids(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--hidden_dim', type=int, default=100, help='hidden dimension, default: 100')
parser.add_argument('--L', type=int, default=2, help='number of hidden layers - 1 (default: 2)')
parser.add_argument('--max_epochs', type=int, default=20000, help='maximum number of epochs to train (default: 20000)')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate (default: 0.05)')
parser.add_argument('--vae_epochs', type=int, default=250, help='number of epochs to train VAE (default: 250)')
parser.add_argument('--vae_lr', type=float, default=0.01, help='learning rate for VAE (default: 0.01)')
parser.add_argument('--balance_lambda', type=float, default=1.0, help='balance parameter (default: 1.0)')
parser.add_argument('--lsd_threshold', type=float, default=2, help='threshold for LSD (default: max 2%), stopping criterion')

args = parser.parse_args()

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_MODE"] = args.wandb_mode

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

data_name = config["dataset"]["data_name"]
print(args)
print('Dataset is:')
print(data_name)

print(json.dumps(config, indent=4))

# Create folder
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if data_name=="acic2016":
    foldername = "./save_diffs4/acic2016/" + "_" + str(args.current_id) + "_" + str(args.nfold) + "/"
elif data_name=="acic2018":
    foldername = "./save_diffs4/acic2018/" + "_" + str(args.current_id) + "_" + str(args.nfold) + "/"
else:
    foldername = "./save_diffs4/" + "_" + str(args.current_id) + "_" + str(args.nfold) + "/"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


current_id = args.current_id
print('Start exe_acic on current_id', current_id)

train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
    dataset_name = data_name,
    current_id = current_id
)

propnet = load_data(dataset_name = data_name, current_id=current_id)
print('Finish training propnet and fix the parameters')
propnet.eval()

device = args.device
if device == "cuda" and not torch.cuda.is_available():
    print("CUDA not available; using CPU.")
    device = "cpu"

propnet = propnet.to(device)
model = CoDiS(config, device).to(device)

if args.smoke_test:
    batch = next(iter(train_loader))
    with torch.no_grad():
        _ = model(batch, is_train=1, propnet=propnet)
    print("SMOKE TEST PASSED: dataloader + propnet + model forward are OK.")
    sys.exit(0)

if "acic2016" in args.config:
    run = wandb.init(
        project=f"CoDiS-acic2016-PaperRevise",
        notes="CoDiS-acic2016,77 files",
        name=f"{args.current_id}_{args.nfold}"
    )
elif "acic2018" in args.config:
    run = wandb.init(
        project="CoDiS-acic2018-PaperRevise",
        notes="CoDiS-acic2018,24 files",
        name=f"{args.current_id}_{args.nfold}"
    )
else:
    run = wandb.init(
        project="CoDiS-ihdp-PaperRevise",
        notes="CoDiS-dataset-ihdp,100 files",
        name=f"{args.current_id}_{args.nfold}"
    )

if args.train:
    wandb.config = {"epochs": config["train"]["epochs"], "num_steps": config["diffusion"]["num_steps"],"lr": config["train"]["lr"]}

    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        valid_epoch_interval=config["train"]["valid_epoch_interval"],
        foldername=foldername,
        propnet = propnet
    )
    print('----------------Finish training------------')
    print('----------------Check trainresults---------')

    print("---------------Start testing---------------")

    evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
    if data_name == 'acic2016':
        directory = "./save_model/acic2016/" + args.current_id + "/" + str(args.nfold)
    elif data_name == 'acic2018':
        directory = "./save_model/acic2018/" + args.current_id + "/" + str(args.nfold)
    else:
        directory = "./save_model/" + args.current_id + "/" + str(args.nfold)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), directory + "/model_weights.pth")
    wandb.finish()

else:
    if data_name == 'acic2016':
        directory = "./save_model/acic2016/" + args.current_id + "/" + str(args.nfold)
    elif data_name == 'acic2018':
        directory = "./save_model/acic2018/" + args.current_id + "/" + str(args.nfold)
    else:
        directory = "./save_model/" + args.current_id + "/" + str(args.nfold)
    model.load_state_dict(torch.load(directory + "/model_weights.pth"))

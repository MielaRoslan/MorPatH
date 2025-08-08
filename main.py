
from configs.configs import build_cfg
from dataset.dataloader import build_dataloaders_from_cfg

def main():
    cfg = build_cfg()
    loaders = build_dataloaders_from_cfg(cfg)
    # model = ...
    # for epoch in range(...):
    #     train_one_epoch(model, loaders["train"], ...)
    #     validate(model, loaders["val"], ...)

if __name__ == "__main__":
    main()

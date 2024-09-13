from argparse import ArgumentParser
from pathlib import Path
import torch
from vilt.datamodules.multitask_datamodule import MTDataModule
from tqdm import tqdm
import random
import numpy as np
import os
from torchvision import transforms
import torch.nn as nn
from datetime import datetime
import wandb
import torch.nn.functional as F
from vilt.gadgets.my_metrics import F1_Score, AUROC, Accuracy, compute_metric
from transformers.optimization import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup
from vilt.utils.heads import get_classifier
from timm.models.vision_transformer import Block
from MetaTransformer.Data2Seq.Data2Seq import Data2Seq
from my_model import My_Meta
import torchvision

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
path = "./"

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Switching to CPU.")

def set_schedule(args, model, train_dataloader):
    lr = args.learning_rate
    wd = args.weight_decay
    lr_mult = args.lr_mult #1
    end_lr = args.end_lr #0
    decay_power = 1

    optimizer_grouped_parameters = [{
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        }, 
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
    )
    max_steps = len(train_dataloader) * args.num_epochs // args.gc
    warmup_steps = args.warmup_steps
    if isinstance(args.warmup_steps, float):
        warmup_steps = int(max_steps * warmup_steps)

    #It combines a warm-up phase with a polynomial decay schedule for the learning rate.
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
        lr_end=end_lr,
        power=decay_power,
    )

    return optimizer, scheduler
    
def train_step(image, text, model, labels, dataset):
    with torch.cuda.amp.autocast():
        logits = model(image, text) #(batch_size, n_classes)
        if (dataset == 'mmimdb'):
            labels=labels.float()
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        elif (dataset == 'Hatefull_Memes' or dataset == 'Food101'):
            labels = labels.long()
            loss = F.cross_entropy(logits, labels)
        else:
            raise NotImplementedError
    return logits, loss

def val_step(image, text, model, labels, dataset):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            logits = model(image, text) #(batch_size, n_classes)
            if (dataset == 'mmimdb'):
                labels=labels.float()
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            elif (dataset == 'Hatefull_Memes' or dataset == 'Food101'):
                labels = labels.long()
                loss = F.cross_entropy(logits, labels)
            else:
                raise NotImplementedError
    return logits, loss

def run(args):
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")    
    training_path: Path = Path(f"{args.data_output_root}/meta/model_{args.dataset}_{args.missing_type}_{args.missing_ratio}_{args.batch_size}_{args.gc}_{args.num_epochs}")

    #Metra Model
    image_tokenizer = Data2Seq(modality='image', dim=768)
    text_tokenizer = Data2Seq(modality='text', dim=768)
    model = My_Meta(image_tokenizer, text_tokenizer, num_classes=args.num_classes, missing_type=args.missing_type, dim=768)
    ckpt = torch.load("MetaTransformer/Meta-Transformer_base_patch16_encoder.pth")
    model.encoder.load_state_dict(ckpt,strict=True)
    model = model.to(device)

    if args.frozen:
        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False


    preprocess=transforms.Compose([
        transforms.Resize(size=224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    dm = MTDataModule(preprocess, preprocess, preprocess, args)  
    dm.setup()
    traindataloader = dm.train_dataloader()
    valdataloader = dm.val_dataloader()
    optimizer, scheduler = set_schedule(args, model, traindataloader)
 
    initial_epoch = 0
    best_metric = float("-inf")
    
    
    if args.resume_training: 
        print('Restoring checkpoint')
        checkpoint = torch.load(args.resume_checkpoint)
        model.image_tokenizer.load_state_dict(checkpoint["image_tokenizer"])
        model.text_tokenizer.load_state_dict(checkpoint["text_tokenizer"])
        model.classifier.load_state_dict(checkpoint["classifier"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        initial_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['best_metric']
        if args.missing_type=='image' or args.missing_type=='text':
            model.cls_token.load_state_dict(checkpoint['cls_token'])
            model.pos_embedding.load_state_dict(checkpoint['pos_embedding'])

    #TRAINING AND VALIDATION
    for epoch in range(initial_epoch, args.num_epochs):
        #TRAINING
        model.train()
        optimizer.zero_grad()
        f1_metric = F1_Score()
        accuracy=Accuracy()
        auroc = AUROC()
        train_bar = tqdm(traindataloader, ncols=150)
        train_running_results = {
            'images_in_epoch': 0, 'accumulated_train_loss': 0}

        for idx, batch in enumerate(train_bar):
            batch_images = batch['image'][0].to(device) #(batch_size, 3, 224, 224)
            batch_descriptions = batch['text'] #list
            labels = torch.tensor(batch['label']).to(device) #(batch_size,) or (bath_size, 23) for mmimdb
            """ 
            #missing_type = 1 = text missing = empty string
            #missing_type = 2 = image missing = image of ones
            #missing_type = 0 = complete
            missingtype = [2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, ...]
            """
            #label, text, text_encodings, image, missing_type
            images_in_batch = batch_images.size(0)

            logits, loss = train_step(batch_images, batch_descriptions, model, labels, args.dataset)
            
            update_train_running_results(
                    train_running_results, loss, images_in_batch)

            #Metrics
            if(args.dataset=='mmimdb'):
                f1_metric.update(logits, labels)
            elif(args.dataset=='Food101'):
                accuracy.update(logits, labels)
            elif args.dataset == 'Hatefull_Memes':
                auroc.update(logits, labels)
            else:
                raise NotImplementedError

            loss = loss/args.gc
            loss.backward()
        
            if((idx + 1) % args.gc == 0) or (idx + 1 == len(traindataloader)):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()       
                wandb.log({"train/lr": scheduler.get_last_lr()[0]})
          
        train_epoch_loss = float(train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
        wandb.log({"train/loss": train_epoch_loss, "train/epoch":epoch})

        if(args.dataset=='mmimdb'):
            F1_Micro, F1_Macro, F1_Samples, F1_Weighted = compute_metric(args, f1_metric)
            metric = F1_Macro.item()
        elif(args.dataset=='Food101'):
            acc = compute_metric(args, accuracy)
            metric = acc.item()
        elif args.dataset == 'Hatefull_Memes':
            aur = compute_metric(args, auroc)
            metric = aur.item()
        else:
            raise NotImplementedError
        wandb.log({"train/metric": metric, "train/epoch_metric":epoch})

        # VALIDATION 
        model.eval()
        f1_metric = F1_Score()
        accuracy=Accuracy()
        auroc = AUROC()
        val_bar = tqdm(valdataloader)
        val_running_results = {
            'images_in_epoch': 0, 'accumulated_val_loss': 0}
        
        for idx, batch in enumerate(val_bar):
            batch_images = batch['image'][0].to(device, non_blocking=True) #(batch_size, 3, 224, 224)
            batch_descriptions = batch['text'] #batch_size
            labels = torch.tensor(batch['label']).float().to(device) #(batch_size, n_classes)

            images_in_batch = batch_images.size(0)
            
            logits, loss = val_step(batch_images, batch_descriptions, model, labels, args.dataset)
            
            update_val_running_results(
                val_running_results, images_in_batch, loss)

            #Metrics
            if(args.dataset=='mmimdb'):
                f1_metric.update(logits, labels)
            elif(args.dataset=='Food101'):
                accuracy.update(logits, labels)
            elif args.dataset == 'Hatefull_Memes':
                auroc.update(logits, labels)
            else:
                raise NotImplementedError
            
        if(args.dataset=='mmimdb'):
            F1_Micro, F1_Macro, F1_Samples, F1_Weighted = compute_metric(args, f1_metric)
            metric = F1_Macro.item()
        elif(args.dataset=='Food101'):
            acc = compute_metric(args, accuracy)
            metric = acc.item()
        elif args.dataset == 'Hatefull_Memes':
            aur = compute_metric(args, auroc)
            metric = aur.item()
        else:
            raise NotImplementedError
        wandb.log({"val/metric": metric, "val/epoch_metric":epoch})

        if metric > best_metric:
            best_metric = metric
            save_model(f'model_{epoch}', epoch, model, optimizer, scheduler, training_start,
                        training_path, best_metric)
                          
        val_epoch_loss = float(val_running_results['accumulated_val_loss'] / val_running_results['images_in_epoch'])
        wandb.log({"val/loss": val_epoch_loss, "val/epoch": epoch})

def update_train_running_results(train_running_results, total_loss, images_in_batch):
    train_running_results["images_in_epoch"] += images_in_batch
    train_running_results['accumulated_train_loss'] += total_loss.float().to('cpu',
                                                                             non_blocking=True).detach() * images_in_batch
      
def update_val_running_results(val_running_results, val_images_in_batch, val_total_loss):
    val_running_results["images_in_epoch"] += val_images_in_batch
    val_running_results['accumulated_val_loss'] += val_total_loss.to('cpu',
                                                                     non_blocking=True).detach().item() * val_images_in_batch
    
def save_model(name, epoch, model_to_save, optimizer, scheduler, training_start, training_path, best_metric):
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)
    if args.missing_type =='both':
        torch.save({
            'epoch': epoch,
            'image_tokenizer': model_to_save.image_tokenizer.state_dict(),
            'text_tokenizer': model_to_save.text_tokenizer.state_dict(),
            'classifier': model_to_save.classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'training_start': training_start,
            'best_metric': best_metric,
        }, str(models_path / f'{name}.pt'))
    elif args.missing_type=='image' or args.missing_type=='text':
        torch.save({
            'epoch': epoch,
            'image_tokenizer': model_to_save.image_tokenizer.state_dict(),
            'text_tokenizer': model_to_save.text_tokenizer.state_dict(),
            'classifier': model_to_save.classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'training_start': training_start,
            'best_metric': best_metric,
            'cls_token': model_to_save.cls_token.data,
            'pos_embedding': model_to_save.pos_embedding.data,
        }, str(models_path / f'{name}.pt'))
    else:
        raise NotImplementedError

if __name__ == '__main__':
    # reproducibility
    torch.manual_seed(0)
    random.seed(0)
    #torch.use_deterministic_algorithms(True)
    np.random.seed(0)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    parser = ArgumentParser()
    parser.add_argument("--num_epochs", default=20, type=int) 
    parser.add_argument("--input-size", default=768, choices=[512, 768], type=int) 
    parser.add_argument("--model", choices=["FCModel", "MLP"], default="MLP", type=str)
    parser.add_argument("--batch_size", default=128, type=int) #256
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--data-root", type=str, default='/work/tesi_asaporita/datasets')  
    parser.add_argument("--data_output_root", type=str, default='/work/tesi_asaporita/checkpoint')
    parser.add_argument("--test-ratio", default=None, type=int)
    parser.add_argument("--test-type", default=None, type=str)
    parser.add_argument("--dataset", default='mmimdb', type=str, choices=["mmimdb", "Hatefull_Memes", "Food101"])
    parser.add_argument("--normalization", default=False, type=bool)
    parser.add_argument("--save-training", default=True, type=bool)
    parser.add_argument("--resume_training", type=bool, help="resume training or not", default=False)
    parser.add_argument("--resume_checkpoint", type=str, help="path to the checkpoint", default="checkpoints/best.pt")
    
    parser.add_argument("--warmup-steps", default=0.1, type=float)
    parser.add_argument("--weight-decay", default=2e-2, type=float) #2e-4
    parser.add_argument("--max-step", default=None, type=int)
    parser.add_argument("--end-lr", default=0, type=float)
    parser.add_argument("--lr-mult", default=1, type=float)
    parser.add_argument("--gc", default=2, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--frozen", default=True, type=bool)


    # missing modality config
    parser.add_argument("--missing_ratio", default=0.7, type=float)
    parser.add_argument("--missing_type", default='both', choices=['text', 'image', 'both'], type=str)
    parser.add_argument("--both-ratio", default=0.5, type=int)
    parser.add_argument("--missing-table-root", default='/work/tesi_asaporita/datasets/missing_tables/', type=str)
    parser.add_argument("--simulate-missing", default=False, type=bool)
    
    # Image setting
    parser.add_argument("--image-size", default=384, type=int)
    parser.add_argument("--max-image-len", default=-1, type=int)
    parser.add_argument("--draw-false-image", default=1, type=int)
    parser.add_argument("--image-only", default=False, type=bool)

    # Text Setting
    parser.add_argument("--vqav2-label-size", default=3129, type=int)
    parser.add_argument("--max-text-len", default=77, type=int)
    parser.add_argument("--draw-false-text", default=0, type=int)

    # Downstream Setting
    parser.add_argument("--get-recall-metric", default=False, type=bool)
    parser.add_argument("--mmimdb-class-num", default=23, type=int)
    parser.add_argument("--hatememes-class-num", default=2, type=int) 
    parser.add_argument("--food101-class-num", default=101, type=int)
    parser.add_argument("--num_classes", default=23, type=int)

    args = parser.parse_args()

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Meta_Missing_Modalities",
        name='MetaTransformer',
        config={
        "model": args.model,
        "learning_rate": args.learning_rate,
        "dataset": args.dataset,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "missing_type":args.missing_type,
        "missing_ratio": args.missing_ratio,
        "gc": args.gc,
        }
    )

    run(args)
    wandb.finish()

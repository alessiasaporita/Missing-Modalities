from argparse import ArgumentParser
from pathlib import Path
import torch
from vilt.datamodules.multitask_datamodule import MTDataModule
from tqdm import tqdm
import random
import numpy as np
import os
from datetime import datetime
import wandb
import torch.nn.functional as F
from vilt.gadgets.my_metrics import F1_Score, AUROC, Accuracy, compute_metric
from transformers.optimization import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup
from vilt.utils.heads import get_classifier
import open_clip


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

def set_schedule(args, classifier, train_dataloader):
    lr = args.learning_rate
    wd = args.weight_decay
    lr_mult = args.lr_mult #1
    end_lr = args.end_lr #0
    decay_power = 1

    optimizer_grouped_parameters = [{
            "params": [
                p
                for n, p in classifier.named_parameters()
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },]

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
    

def run(args):
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(f"{args.data_output_root}/open_clip/classifier/{args.clip_model}/classifier_{args.model}_{args.dataset}_{args.fusion}_{args.missing_type}_{args.missing_ratio}_{args.batch_size}_{args.gc}")
    
    #Open CLIP
    if args.clip_model == 'RN50-CC12M':
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:thaottn/OpenCLIP-resnet50-CC12M')
        tokenizer = open_clip.get_tokenizer('hf-hub:thaottn/OpenCLIP-resnet50-CC12M')
    elif args.clip_model=='RN50-YFCC15M':   
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:thaottn/OpenCLIP-resnet50-YFCC15M')
        tokenizer = open_clip.get_tokenizer('hf-hub:thaottn/OpenCLIP-resnet50-YFCC15M')
    else:
        NotImplementedError

    dm = MTDataModule(preprocess_val, preprocess_val, preprocess_val, args) #lista 
    dm.setup()
    traindataloader = dm.train_dataloader()
    valdataloader = dm.val_dataloader()
    classifier = get_classifier(args, device)
    optimizer, scheduler = set_schedule(args, classifier, traindataloader)
 
    initial_epoch = 0
    best_metric = float("-inf")
    model = model.to(device, non_blocking=True)
    model.eval()

    #TRAINING AND VALIDATION
    for epoch in range(initial_epoch, args.num_epochs):
        #TRAINING
        classifier.train()
        train_bar = tqdm(traindataloader, ncols=150)
        train_running_results = {
            'images_in_epoch': 0, 'accumulated_train_loss': 0}
        
        for idx, batch in enumerate(train_bar):
            batch_images = batch['image'][0] #(batch_size, 3, 224, 224)
            batch_descriptions = batch['text'] #batch_size
            """ 
            #missing_type = 1 = text missing = empty string
            #missing_type = 2 = image missing = image of ones
            #missing_type = 0 = complete
            missingtype = [2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, ...]
            """
            #label, text, image, missing_type

            batch_images = batch_images.to(device, non_blocking=True)
            images_in_batch = batch_images.size(0)
            batch_texts = tokenizer(
                batch_descriptions).to(device, non_blocking=True) #(batch_size, 77)
            
            with torch.no_grad():
                image_features = model.encode_image(batch_images) #(batch_size, 1024)
                text_features = model.encode_text(batch_texts) #(batch_size, 1024)
            
            # normalized features
            if args.normalization:
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

            if(args.fusion == 'average'):
                if args.missing_type == 'both':
                    input_model = (image_features + text_features) / 2.0
                elif args.missing_type == 'text':
                    input_model = image_features 
                elif args.missing_type == 'image':
                    input_model = text_features 
                else:
                    raise NotImplementedError
            elif args.fusion == 'sum':
                if args.missing_type == 'both':
                    input_model = image_features+text_features
                elif args.missing_type == 'text':
                    input_model = image_features * 2.0
                elif args.missing_type == 'image':
                    input_model = text_features * 2.0
                else:
                    raise NotImplementedError
            else:
                    raise NotImplementedError
           

            labels = torch.tensor(batch['label']).to(device) #(batch_size,) or (bath_size, 23) for mmimdb

            with torch.cuda.amp.autocast():
                logits = classifier(input_model) #(batch_size, n_classes) #logits float_16
                if (args.dataset == 'mmimdb'):
                    labels=labels.float()
                    loss = F.binary_cross_entropy_with_logits(logits, labels)
                elif (args.dataset == 'Hatefull_Memes' or args.dataset == 'Food101'):
                    labels = labels.long()
                    loss = F.cross_entropy(logits, labels)
                else:
                    raise NotImplementedError
            update_train_running_results(
                    train_running_results, loss, images_in_batch)
            wandb.log({"train/batch_loss": loss})

            loss = loss/args.gc
            loss.backward()
        
            if((idx + 1) % args.gc == 0) or (idx + 1 == len(traindataloader)):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()            
                wandb.log({"train/lr": scheduler.get_last_lr()[0]})
          
        train_epoch_loss = float(train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
        wandb.log({"train/loss": train_epoch_loss, "train/epoch":epoch})

        # VALIDATION 
        classifier.eval()
        f1_metric = F1_Score()
        accuracy=Accuracy()
        auroc = AUROC()
        val_bar = tqdm(valdataloader)
        val_running_results = {
            'images_in_epoch': 0, 'accumulated_val_loss': 0}
        
        for idx, batch in enumerate(val_bar):
            batch_images = batch['image'][0] #(batch_size, 3, 224, 224)
            batch_descriptions = batch['text'] #batch_size

            batch_images = batch_images.to(device, non_blocking=True)
            images_in_batch = batch_images.size(0)
            batch_texts = tokenizer(
                batch_descriptions).to(device, non_blocking=True) #(batch_size, 77)
            
            with torch.no_grad():
                image_features = model.encode_image(batch_images) #(batch_size, 1024)
                text_features = model.encode_text(batch_texts) #(batch_size, 1024)

                # normalized features
                if args.normalization:
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)

                if(args.fusion == 'average'):
                    if args.missing_type == 'both':
                        input_model = (image_features + text_features) / 2.0
                    elif args.missing_type == 'text':
                        input_model = image_features 
                    elif args.missing_type == 'image':
                        input_model = text_features 
                    else:
                        raise NotImplementedError
                elif args.fusion == 'sum':
                    if args.missing_type == 'both':
                        input_model = image_features+text_features
                    elif args.missing_type == 'text':
                        input_model = image_features * 2.0
                    elif args.missing_type == 'image':
                        input_model = text_features * 2.0
                    else:
                        raise NotImplementedError
                else:
                        raise NotImplementedError

                labels = torch.tensor(batch['label']).float().to(device) #(batch_size, n_classes)

                with torch.cuda.amp.autocast():
                    logits = classifier(input_model) #(batch_size, n_classes)
                    if (args.dataset == 'mmimdb'):
                        labels=labels.float()
                        loss = F.binary_cross_entropy_with_logits(logits, labels)
                    elif (args.dataset == 'Hatefull_Memes' or args.dataset == 'Food101'):
                        labels = labels.long()
                        loss = F.cross_entropy(logits, labels)
                    else:
                        raise NotImplementedError

                update_val_running_results(
                    val_running_results, images_in_batch, loss)
                wandb.log({"val/batch_loss": loss})

                loss = loss / args.gc

                #Metrics
                if(args.dataset=='mmimdb'):
                    f1_metric.update(logits, labels)
                elif(args.dataset=='Food101'):
                    accuracy.update(logits, labels)
                elif args.dataset == 'Hatefull_Memes':
                    auroc.update(logits, labels)
                else:
                    raise NotImplementedError
            
        if args.save_training:
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

            if metric > best_metric:
                best_metric = metric
                save_model(f'classifier_{epoch}', epoch, classifier, optimizer, scheduler, training_start,
                            training_path)
                wandb.log({"val/best_metric": best_metric, "val/epoch_best_metric":epoch})
                
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

def save_model(name, epoch, model_to_save, optimizer, scheduler, training_start, training_path):
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)
    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': epoch,
        model_name: model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'training_start': training_start,
    }, str(models_path / f'{name}.pt'))


if __name__ == '__main__':
    # reproducibility
    torch.manual_seed(0)
    random.seed(0)
    #torch.use_deterministic_algorithms(True)
    np.random.seed(0)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    parser = ArgumentParser()
    parser.add_argument("--num-epochs", default=20, type=int) 
    parser.add_argument("--input-size", default=1024, choices=[512, 768, 1024], type=int) 
    parser.add_argument("--model", default="MLP", type=str)
    parser.add_argument("--batch-size", default=128, type=int) #256
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--data-root", type=str, default='/work/tesi_asaporita/datasets')
    parser.add_argument("--data-output-root", type=str, default='/work/tesi_asaporita/checkpoint')
    parser.add_argument("--test-ratio", default=None, type=int)
    parser.add_argument("--test-type", default=None, type=str)
    parser.add_argument("--dataset", default='Food101', type=str, choices=["mmimdb", "Hatefull_Memes", "Food101"])
    parser.add_argument("--fusion", default='sum', choices=['sum', 'average'], type=str) 
    parser.add_argument("--normalization", default=True, type=bool)
    parser.add_argument("--save-training", default=True, type=bool)
    parser.add_argument("--clip-model", default='RN50-CC12M', type=str, choices=['RN50-CC12M', 'RN50-YFCC15M'])

    parser.add_argument("--warmup-steps", default=0.1, type=float)
    parser.add_argument("--weight-decay", default=2e-2, type=float)
    parser.add_argument("--max-step", default=None, type=int)
    parser.add_argument("--end-lr", default=0, type=float)
    parser.add_argument("--lr-mult", default=1, type=float)
    parser.add_argument("--gc", default=2, type=int)
    parser.add_argument("--num_workers", default=2, type=int)


    # missing modality config
    parser.add_argument("--missing-ratio", default=0.7, type=float)
    parser.add_argument("--missing-type", default='both', choices=['text', 'image', 'both'], type=str)
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

    args = parser.parse_args()

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="OpenClip",
        name='RN50-CC12M',
        config={
        "model": args.model,
        "learning_rate": args.learning_rate,
        "dataset": args.dataset,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "fusion": args.fusion,
        "missing_type":args.missing_type,
        "missing_ratio": args.missing_ratio,
        "gc": args.gc,
        }
    )

    run(args)

    wandb.finish()

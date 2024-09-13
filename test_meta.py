from argparse import ArgumentParser
from pathlib import Path
import torch
from vilt.datamodules.multitask_datamodule import MTDataModule
from tqdm import tqdm
import random
import numpy as np
import os
import json
from vilt.gadgets.my_metrics import F1_Score, AUROC, Accuracy, compute_metric
from torchvision import transforms
from MetaTransformer.Data2Seq.Data2Seq import Data2Seq
from my_model import My_Meta

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
   

def test(args):
    path: Path = Path(f"{args.root}/meta/model_{args.dataset}_{args.missing_type}_{args.missing_ratio}_{args.batch_size}_{args.gc}_{args.num_epochs}/saved_models/{args.best_model}")
    path_metric: Path = Path(f'{args.data_output_root}metrics/meta/meta_{args.dataset}_{args.missing_type}_{args.missing_ratio}_{args.batch_size}_{args.gc}_{args.num_epochs}')
    path_metric.mkdir(exist_ok=True, parents=True)
    
    #Open CLIP
    image_tokenizer = Data2Seq(modality='image', dim=768)
    text_tokenizer = Data2Seq(modality='text', dim=768)
    model = My_Meta(image_tokenizer, text_tokenizer, num_classes=args.num_classes, missing_type=args.missing_type, dim=768)
    ckpt = torch.load("MetaTransformer/Meta-Transformer_base_patch16_encoder.pth")
    model.encoder.load_state_dict(ckpt,strict=True)
    model = model.to(device)

    preprocess=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dm = MTDataModule(preprocess, preprocess, preprocess, args) #lista 
    dm.setup()
    testdaloader = dm.test_dataloader()

    checkpoint = torch.load(path, map_location=device)
    model.image_tokenizer.load_state_dict(checkpoint["image_tokenizer"])
    model.text_tokenizer.load_state_dict(checkpoint["text_tokenizer"])
    model.classifier.load_state_dict(checkpoint["classifier"])

    # TEST 
    f1_metric = F1_Score()
    accuracy=Accuracy()
    auroc = AUROC()
    model.eval()
    
    test_bar = tqdm(testdaloader)
    
    for batch in test_bar:
        batch_images = batch['image'][0].to(device) #(batch_size, 3, 224, 224)
        batch_descriptions = batch['text'] #batch_size
        labels = torch.tensor(batch['label']).to(device)
      
        with torch.no_grad():
            logits = model(batch_images, batch_descriptions)

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
        scores = {'F1_Macro': F1_Macro.item()}
    elif(args.dataset=='Food101'):
        acc = compute_metric(args, accuracy)
        scores = {'acc': acc.item()}
    elif args.dataset == 'Hatefull_Memes':
        aur = compute_metric(args, auroc)
        scores = {'aur': aur.item()}
    else:
        raise NotImplementedError
    filename = os.path.join(path_metric, "metric.txt")
    with open(filename, 'w') as f:
        f.write(json.dumps(scores))
    

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
    parser.add_argument("--model", choices=["FCModel", "MLP"], default="MLP", type=str)
    parser.add_argument("--batch-size", default=128, type=int) #256
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--data-root", type=str, default='/work/tesi_asaporita/datasets')
    parser.add_argument("--root", type=str, default='/work/tesi_asaporita/checkpoint')
    parser.add_argument("--data-output-root", type=str, default='./')
    parser.add_argument("--test-ratio", default=None, type=int)
    parser.add_argument("--test-type", default=None, type=str)
    parser.add_argument("--dataset", default='Food101', type=str, choices=["mmimdb", "Hatefull_Memes", "Food101"])
    parser.add_argument("--normalization", default=True, type=bool)
    parser.add_argument("--best_model", default='model_15.pt', type=str)
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
    parser.add_argument("--num_classes", default=101, type=int)

    args = parser.parse_args()

    test(args)

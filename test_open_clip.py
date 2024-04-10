from argparse import ArgumentParser
from pathlib import Path
import torch
from vilt.datamodules.multitask_datamodule import MTDataModule
from tqdm import tqdm
import random
import numpy as np
import os
import open_clip
import json
from vilt.gadgets.my_metrics import F1_Score, AUROC, Accuracy, compute_metric
from vilt.utils.heads import get_classifier

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
    path: Path = Path(f"{args.root}/open_clip/classifier/{args.clip_model}/classifier_{args.model}_{args.dataset}_{args.fusion}_{args.missing_type}_{args.missing_ratio}_{args.batch_size}_{args.gc}/saved_models/{args.best_classifier}")
    path_metric: Path = Path(f'{args.data_output_root}metrics/open_clip/{args.clip_model}/{args.model}_{args.dataset}_{args.fusion}_{args.missing_type}_{args.missing_ratio}_{args.batch_size}_{args.gc}')
    path_metric.mkdir(exist_ok=True, parents=True)
    
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
    testdaloader = dm.test_dataloader()

    classifier = get_classifier(args, device)
    state_dict = torch.load(path, map_location=device)
    classifier.load_state_dict(state_dict[args.model])

    # TEST 
    f1_metric = F1_Score()
    accuracy=Accuracy()
    auroc = AUROC()
    classifier.eval()
    model = model.to(device, non_blocking=True)
    model.eval()
    test_bar = tqdm(testdaloader)
    
    for batch in test_bar:
        batch_images = batch['image'][0] #(batch_size, 3, 224, 224)
        batch_descriptions = batch['text'] #batch_size

        batch_images = batch_images.to(device, non_blocking=True)
        
        batch_texts = tokenizer(
            batch_descriptions).to(device, non_blocking=True) #(batch_size, 77)
        
        with torch.no_grad():
            image_features = model.encode_image(batch_images) #(batch_size, 512)
            text_features = model.encode_text(batch_texts) #(batch_size, 512)

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

            labels = torch.tensor(batch['label']).to(device) #(batch_size, 23)/(batch_size,)

            with torch.cuda.amp.autocast():
                logits = classifier(input_model) #(batch_size, n_classes)

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
    parser.add_argument("--data-root", type=str, default='/work/tesi_asaporita/MissingModalities/datasets')
    parser.add_argument("--root", type=str, default='/work/tesi_asaporita/MissingModalities')
    parser.add_argument("--data-output-root", type=str, default='./')
    parser.add_argument("--test-ratio", default=None, type=int)
    parser.add_argument("--test-type", default=None, type=str)
    parser.add_argument("--dataset", default='Food101', type=str, choices=["mmimdb", "Hatefull_Memes", "Food101"])
    parser.add_argument("--fusion", default='sum', choices=['sum', 'average'], type=str)
    parser.add_argument("--normalization", default=True, type=bool)
    parser.add_argument("--best-classifier", default='classifier_5.pt', type=str)
    parser.add_argument("--clip-model", default='RN50-CC12M', type=str, choices=['RN50-CC12M', 'RN50-YFCC15M'])
    parser.add_argument("--gc", default=2, type=int)
    parser.add_argument("--num_workers", default=2, type=int)

    # missing modality config
    parser.add_argument("--missing-ratio", default=0.7, type=float)
    parser.add_argument("--missing-type", default='text', choices=['text', 'image', 'both'], type=str)
    parser.add_argument("--both-ratio", default=0.5, type=int)
    parser.add_argument("--missing-table-root", default='/work/tesi_asaporita/MissingModalities/datasets/missing_tables/', type=str) 
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

    test(args)

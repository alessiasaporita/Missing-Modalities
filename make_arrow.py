import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mmimdb', type=str, help='Datasets.')
parser.add_argument('--root', default='/work/tesi_asaporita/MissingModalities/', type=str, help='Root of datasets')
args = parser.parse_args()


if args.dataset.lower() == 'mmimdb':
    from vilt.utils.write_mmimdb import make_arrow
    make_arrow(f'{args.root}data/mmimdb', f'{args.root}datasets/mmimdb')
    
elif args.dataset.lower() == 'food101':
    from vilt.utils.write_food101 import make_arrow
    make_arrow(f'{args.root}data/Food101', f'{args.root}datasets/Food101')
    
elif args.dataset.lower() == 'hateful_memes':
    from vilt.utils.write_hatememes import make_arrow
    make_arrow(f'{args.root}data/Hatefull_Memes', f'{args.root}datasets/Hatefull_Memes')


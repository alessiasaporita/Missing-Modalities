import jsonlines
import pandas as pd
import pyarrow as pa
import os

from tqdm import tqdm

def make_arrow(root, dataset_root, single_plot=False):
    split_sets = ['train', 'dev', 'test']
    
    for split in split_sets:
        data_list = []
        with jsonlines.open(os.path.join(root,f'{split}.jsonl'), 'r') as rfd: #./data/Hatefull_Memes/data/train.jsonl
            for data in tqdm(rfd): #id, img, label, text
                image_path = os.path.join(root, data['img'])
                
                with open(image_path, "rb") as fp:
                    binary = fp.read()       
                    
                text = [data['text']]
                label = data['label']
                #text_aug = text_aug_dir['{}.png'.format(data['id'])]

                data = (binary, text, label, split)
                data_list.append(data)                
                            

        dataframe = pd.DataFrame(
            data_list, #list of 8500/500/1000 (img, text, label, split='train', 'dev', 'test')
            columns=[
                "image",
                "text",
                "label",
                "split",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/hatememes_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)        
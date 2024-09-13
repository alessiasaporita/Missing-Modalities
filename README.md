## Multimodal Models with Missing Modalities for Visual Recognition
In this code, we analyze transformer-based models' robustness in multimodal learning for visual recognition when missing-modality occurs either during training or testing in real-world situations. To this end, we freeze the model's parameters and train only the final classifier. 
This is the code for the multimodal classification task using MM-IMDb, UPMC Food101, and Hateful Memes with image and text modalities. 


### Environment
* Python 3.8.5
* torch 1.9.0+cu111
* torchaudio 0.9.0
* torchvision 0.10.0+cu111

#### Other requirements
```
environment.yml
```

### Dataset
We use three vision and language datasets: [MM-IMDb](https://github.com/johnarevalo/gmu-mmimdb), [UPMC Food-101](https://visiir.isir.upmc.fr/explore), and [Hateful Memes](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/). We use `pyarrow` to serialize the datasets, the conversion codes are located in `vilt/utils/write_*.py`. Please see `DATA.md` to organize the datasets, otherwise you may need to revise the `write_*.py` files to meet your dataset path and files. Run the following script to create the pyarrow binary file:
```
python make_arrow.py --dataset [DATASET] --root [YOUR_DATASET_ROOT]
```

### Run Demo
* To run the code:
```
python main_clip.py
```
```
python main_open_clip.py
```
```
python main_meta.py
```




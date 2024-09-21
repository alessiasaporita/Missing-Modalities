import torch
from torch import nn
from einops import rearrange, repeat
from timm.models.vision_transformer import Block
from MetaTransformer.Data2Seq.Data2Seq import Data2Seq
from MetaTransformer.Data2Seq.Image import get_image_embeddings
from MetaTransformer.Data2Seq.Text import get_text_embeddings

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class MetaTransformer(nn.Module): #dim=768 
    def __init__(self, num_classes: int, missing_type: str ='both', dim: int = 768, pretrained_model: str = 'mae', pooling: str = 'avg', fusion: str = 'concat'):
        super().__init__()

        self.missing_type = missing_type
        self.pretrained_model = pretrained_model
        self.dim = dim
        self.pooling = pooling
        self.fusion = fusion

        #Pretrained tokenizer
        self.image_tokenizer = Data2Seq(modality='image', dim=768) 
        self.text_tokenizer = Data2Seq(modality='text', dim=768)

        self.position_embedding = nn.Parameter(torch.zeros(1, self.image_tokenizer.embed.num_patches + 1, dim)) #(1, N+1, D) 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))                                                   #(1, N, D)  

        #Froozen Encoder                 
        self.encoder = nn.Sequential(*[ 
                    Block(
                        dim=768,
                        num_heads=12,
                        mlp_ratio=4.,
                        qkv_bias=True,
                        norm_layer=nn.LayerNorm,
                        act_layer=nn.GELU
                    )
                    for i in range(12)])

        #Classifier
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, num_classes),
        )
        self.init_weights(self.pretrained_model)
        
    def init_weights(self, pretrained_model):
        #Text Tokenizer Inizialization
        with torch.no_grad():
            checkpoint = torch.load("pretrained_models/clip-vit-large-patch14-embedding.pt")
            self.text_tokenizer.embed.embed.weight.copy_(checkpoint['weight'])
            
            #Image Tokenizer Inizialization
            if pretrained_model=='clip':
                checkpoint = torch.load("pretrained_models/clip-vit-base-patch16-emb.pt")
                self.image_tokenizer.embed.proj.weight.copy_(checkpoint['patch_embedding.weight']) 
                self.image_tokenizer.embed.proj.bias.data.zero_()
                self.image_tokenizer.embed.proj.bias.requires_grad = False #disable the bias term
            elif pretrained_model=='vit':
                checkpoint = torch.load("pretrained_models/vit-base-patch16-224-emb.pt")
                self.image_tokenizer.embed.proj.weight.copy_(checkpoint['patch_embeddings.projection.weight'])
                self.image_tokenizer.embed.proj.bias.copy_(checkpoint['patch_embeddings.projection.bias'])
                self.position_embedding.copy_(checkpoint['position_embeddings'])
                self.cls_token.copy_(checkpoint['cls_token'])
            elif pretrained_model=='mae':
                checkpoint = torch.load("pretrained_models/vit_base_patch16_224-mae-emb.pt")
                self.image_tokenizer.embed.proj.weight.copy_(checkpoint['proj.weight'])
                self.image_tokenizer.embed.proj.bias.copy_(checkpoint['proj.bias'])
            else:
                raise NotImplementedError
        
            #Encoder inizialiazion
            ckpt = torch.load("MetaTransformer/Meta-Transformer_base_patch16_encoder.pth")
            self.encoder.load_state_dict(ckpt,strict=True)
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            #Classifier inizialiazion
            if isinstance(self.classifier, (nn.Linear, nn.Embedding)):
                self.classifier.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(self.classifier, nn.LayerNorm):
                self.classifier.bias.data.zero_()
                self.classifier.weight.data.fill_(1.0)

            if isinstance(self.classifier, nn.Linear) and self.classifier.bias is not None:
                self.classifier.bias.data.zero_()

    def forward(self, image, text):
        #Extraction of features
        image_features = get_image_embeddings(image).float()    #(B, N, D) 
        text_features, text_tensor = get_text_embeddings(text)
        text_features = text_features.float()                   #(B, n, D)
        #text_features = self.text_tokenizer(text)   #(B, n, D)
        #image_features = self.image_tokenizer(image) #(B, N, D)
        
        #Unimodal/Multimodal Input
        if self.fusion == 'concat':
            if self.missing_type == 'both':
                features = torch.cat([image_features, text_features],dim=1) #(B, N+n, D)
            elif self.missing_type == 'text':
                features = image_features
            elif self.missing_type == 'image':
                features = text_features
            else:
                raise NotImplementedError
        elif self.fusion=='cls_pos':
            if self.missing_type == 'both':
                pass #I don't know
            elif self.missing_type == 'text':
                b, n, _ = image_features.shape
                cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
                features = torch.cat((cls_tokens, image_features), dim=1) + self.position_embedding
            elif self.missing_type == 'image':
                seq_length = text_features.shape[-2] + 1
                position_ids = self.position_embedding[:, :seq_length]
                b, n, _ = text_features.shape
                cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
                features = torch.cat((cls_tokens, text_features), dim=1) + position_ids
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        #Extraction of semantic embeddings
        x = self.encoder(features)  # (B, seq_len, D)
        
        #Pooling (Global Average Pooling/CLS Output) (B, D)
        if self.pooling == 'avg':
            pooled_output = x.mean(1)      
        elif self.pooling == 'cls':
            if self.missing_type == 'text':
                pooled_output = x[:, 0, :] 
            elif self.missing_type == 'image':
                pooled_output = x[torch.arange(x.shape[0]), text_tensor.argmax(dim=-1)]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError 

        #Head classification
        x = self.classifier(pooled_output)

        return x
    
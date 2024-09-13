import torch
from torch import nn
from transformers.models.clip import CLIPModel
from einops import rearrange, repeat
from timm.models.vision_transformer import Block

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class My_Meta(nn.Module): #dim=768 
    def __init__(self, image_tokenizer, text_tokenizer, num_classes, missing_type='both', dim=768, text_pos = 77, image_pos = 196):
        super().__init__()

        self.image_tokenizer = image_tokenizer
        self.text_tokenizer = text_tokenizer
        
        if missing_type == 'image' or missing_type=='text':
            num_position = max(text_pos, image_pos)
            self.pos_embedding = nn.Parameter(torch.randn(1, num_position + 1, dim)) #(1, N+n, D)
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) #(1, 1, D)
            
        #self.text_pos_embedding = nn.Parameter(torch.randn(1, text_pos + 1, dim)) #(1, n, D)
        #self.image_pos_embedding = nn.Parameter(torch.randn(1, image_pos + 1, dim)) #(1, N, D)

        #self.image_cls_token = nn.Parameter(torch.randn(1, 1, dim)) #(1, 1, D)
        #self.text_cls_token = nn.Parameter(torch.randn(1, 1, dim)) #(1, 1, D)

        self.missing_type = missing_type
        
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

        self.classifier = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, num_classes),
        )
        init_weights(self.classifier)
        

    def forward(self, image, text):
        #Extraction of embedding
        image_features = self.image_tokenizer(image) #(B, N, D)
        text_features = self.text_tokenizer(text)  #(B, n, D)
        
        #Unimodal/Multimodal Input
        if self.missing_type == 'both':
            features = torch.cat([image_features, text_features],dim=1) #(B, N+n, D)
        elif self.missing_type == 'text':
            seq_length = image_features.shape[-2] + 1
            b, n, _ = image_features.shape
            position_ids = self.pos_embedding[:, :seq_length]
            cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
            features = torch.cat((cls_tokens, image_features), dim=1) + position_ids
        elif self.missing_type == 'image':
            seq_length = text_features.shape[-2] + 1
            b, n, _ = text_features.shape
            position_ids = self.pos_embedding[:, :seq_length]
            cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
            features = torch.cat((cls_tokens, text_features), dim=1) + position_ids
        else:
            raise NotImplementedError
        
        with torch.no_grad():
            #features = features.permute(1, 0, 2)  # NLD -> LND
            x = self.encoder(features) 
            #x = x.permute(1, 0, 2)  # LND -> NLD

        #Pooling
        if self.missing_type == 'both':
            pooled_output = x.mean(1) #(B, D)
        elif self.missing_type == 'text' or self.missing_type=='image':
            pooled_output = x[:, 0, :] #(B, D)
        else:
            raise NotImplementedError

        #Head classification
        x = self.classifier(pooled_output)

        return x
    """
    def forward(self, image, text):
        image_features = self.image_tokenizer(image) #(B, N, D)
        text_features = self.text_tokenizer(text)  #(B, n, D)
        seq_length = text_features.shape[-2] + 1
        position_ids = self.text_pos_embedding[:, :seq_length]

        b, n, _ = image_features.shape
        image_cls_tokens = repeat(self.image_cls_token, '1 n d -> b n d', b = b)
        text_cls_tokens = repeat(self.text_cls_token, '1 n d -> b n d', b = b)
        x_image = torch.cat((image_cls_tokens, image_features), dim=1) + self.image_pos_embedding #(B, N+1, D)
        x_text = torch.cat((text_cls_tokens, text_features), dim=1) + position_ids #(B, n+1, D)

        with torch.no_grad():
            x_image = self.encoder(x_image) #(B, N+1, D)
            x_text = self.encoder(x_text) #(B, n+1, D)

        x_image = x_image[:, 0] #(B, D)
        x_text = x_text[:, 0] #(B, D)

        if self.missing_type == 'both':
            x = x_image + x_text
        elif self.missing_type == 'text':
            x = x_image 
        elif self.missing_type == 'image':
            x = x_text 
        else:
            raise NotImplementedError
        
        x = self.classifier(x)

        return x
    """
    
    
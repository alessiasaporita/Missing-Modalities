import torch
import torch.nn as nn
import torch
from torch import nn
from timm.models.vision_transformer import Block
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

class MetaModel(nn.Module):
    def __init__(self, num_classes: int, missing_type: str ='both'):
        super().__init__()

        #load encoder
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
        
        ckpt = torch.load("MetaTransformer/Meta-Transformer_base_patch16_encoder.pth")
        self.encoder.load_state_dict(ckpt,strict=True)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # load clip
        self.clip, self.preprocess = clip.load('ViT-L/14', device)
        for param in self.clip.parameters():
            param.requires_grad = False        

        self.clip_width = self.clip.visual.proj.shape[1] #self.clip.visual.conv1.out_channels
        self.seq_length = 256 + 1
        self.missing_type = missing_type

        #load classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.clip_width, self.clip_width * 2),
            nn.LayerNorm(self.clip_width * 2),
            nn.GELU(),
            nn.Linear(self.clip_width * 2, num_classes),
        )
        if isinstance(self.classifier, (nn.Linear, nn.Embedding)):
            self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(self.classifier, nn.LayerNorm):
            self.classifier.bias.data.zero_()
            self.classifier.weight.data.fill_(1.0)
        if isinstance(self.classifier, nn.Linear) and self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
        
        for name, param in self.named_parameters():
            if param.requires_grad:
               print(f"Trainable param: {name}, {param.shape}, {param.dtype}")

        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameter count : {count}")

    def encode_image(self, x):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                    x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        pos_embedding = self.clip.visual.positional_embedding
        
        x = x + pos_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x

    def encode_text(self, text):
        text_tensor = clip.tokenize(text, truncate=True).to(device) #(B, 77)
        x = self.clip.token_embedding(text_tensor).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip.positional_embedding.type(self.clip.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x).type(self.clip.dtype)
        x = x @ self.clip.text_projection

        return x, text_tensor
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        #x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection

    def forward(self, image, text):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                image_features = self.encode_image(image)      
                text_features, text_tensor = self.encode_text(text)
        
        text_features = text_features.float()                   #(B, n=77, D)
        image_features = image_features.float()                 #(B, N+1=257, D)

        if self.missing_type == 'both':
            x = self.encoder(image_features)  # (B, seq_len, D)
            pooled_output_image = x[:, 0, :] 
            x = self.encoder(text_features)  # (B, seq_len, D)
            pooled_output_text = x[torch.arange(x.shape[0]), text_tensor.argmax(dim=-1)]
            
            #pooled_output_text = pooled_output_text / pooled_output_text.norm(dim=1, keepdim=True)
            #pooled_output_image = pooled_output_image / pooled_output_image.norm(dim=1, keepdim=True)
            pooled_output = (pooled_output_text + pooled_output_image) / 2.0

        elif self.missing_type == 'text':
            x = self.encoder(image_features)  # (B, seq_len, D)
            pooled_output = x[:, 0, :] 

        elif self.missing_type == 'image':
            x = self.encoder(text_features)  # (B, seq_len, D)
            pooled_output = x[torch.arange(x.shape[0]), text_tensor.argmax(dim=-1)]

        else:
            raise NotImplementedError                    

        x = self.classifier(pooled_output)
        
        return x
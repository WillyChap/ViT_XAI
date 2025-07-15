import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
ViT Code Credit (29 April 2025): 
    https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6

Updated by Kirsten Mayer (29 April 2025)
'''

def dense(in_features, out_features, act_fun=True, reshape_shape=(18,36), *args, **kwargs):

        return torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True),
            getattr(torch.nn, act_fun)(),
            nn.Unflatten(1, (reshape_shape))
        )
        
def conv_couplet(in_channels, out_channels, act_fun, *args, **kwargs):
    if act_fun == "Linear":
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            torch.nn.BatchNorm2d(out_channels),
        )
    else:
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            getattr(torch.nn, act_fun)(),
            torch.nn.BatchNorm2d(out_channels),
        )


def upconv_couplet(in_channels, out_channels, act_fun, *args, **kwargs):
    if act_fun=="Linear":
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, *args, **kwargs),
            torch.nn.BatchNorm2d(out_channels)
        )
    else:
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, *args, **kwargs),
            getattr(torch.nn, act_fun)(),
            torch.nn.BatchNorm2d(out_channels),
        )

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        
        return x


class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size
    
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x):
        # Obtaining Queries, Keys, and Values
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
    
        # Dot Product of Queries and Keys
        attention = Q @ K.transpose(-2,-1)
    
        # Scaling
        attention = attention / (self.head_size ** 0.5)
    
        attention = torch.softmax(attention, dim=-1)
    
        attention = attention @ V
    
        return attention
        

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.head_size = d_model // n_heads
    
        self.W_o = nn.Linear(d_model, d_model)
    
        self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])

    def forward(self, x):
        # Combine attention heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)
    
        out = self.W_o(out)
    
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # Classification Token
        
        # Creating positional encoding
        pe = torch.zeros(max_seq_length, d_model)
    
        for pos in range(max_seq_length):
          for i in range(d_model):
            if i % 2 == 0:
              pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
            else:
              pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        # Expand to have class token for every image in batch
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)
        
        # Adding class tokens to the beginning of each embedding
        x = torch.cat((tokens_batch, x), dim=1)
        
        # Add positional encoding to embeddings
        x = x + self.pe
        
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()
        
        self.d_model = d_model # Dimensionality of Model
        self.img_size = img_size # Image Size
        self.patch_size = patch_size # Patch Size
        self.n_channels = n_channels # Number of Channels
        
        self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)
        
    # B: Batch Size
    # C: Image Channels
    # H: Image Height
    # W: Image Width
    # P_col: Patch Column
    # P_row: Patch Row
    def forward(self, x):
        x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)
        
        x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)
        
        x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model)
    
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
    
        # Sub-Layer 1 Normalization
        self.ln1 = nn.LayerNorm(d_model)
    
        # Multi-Head Attention
        self.mha = MultiHeadAttention(d_model, n_heads)
    
        # Sub-Layer 2 Normalization
        self.ln2 = nn.LayerNorm(d_model)
    
        # Multilayer Perception
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*r_mlp),
            nn.GELU(),
            nn.Linear(d_model*r_mlp, d_model)
        )

    def forward(self, x):
        # Residual Connection After Sub-Layer 1
        out = x + self.mha(self.ln1(x))
    
        # Residual Connection After Sub-Layer 2
        out = out + self.mlp(self.ln2(out))
    
        return out
      

class VisionTransformer(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels, n_heads, n_layers, decoder_config):
        super().__init__()
    
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    
        self.d_model = d_model # Dimensionality of model
        self.img_size = img_size # Image size
        self.patch_size = patch_size # Patch size
        self.n_channels = n_channels # Number of channels
        self.n_heads = n_heads # Number of attention heads
    
        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
        self.max_seq_length = self.n_patches + 1
    
        self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(self.d_model, self.n_heads) for _ in range(n_layers)])

        self.decoder_config = decoder_config
    
        # Decoder
        self.upsamp1 = UpBlock(
            in_channels=decoder_config["filters"][0],
            out_channels=decoder_config["filters"][1])

        self.upsamp2 = UpBlock(
            in_channels=decoder_config["filters"][1],
            out_channels=decoder_config["filters"][2])
    
        # self.dense1 = dense(
        #     in_features=decoder_config["hiddens_final_in"],
        #     out_features=decoder_config["hiddens_final_out"][0],
        #     act_fun=decoder_config["hiddens_act_func"][0],
        #     reshape_shape=decoder_config["reshape_shape"]
        #     )

        
        # self.upconv1 = upconv_couplet(
        #     in_channels=decoder_config["filters"][0],
        #     out_channels=decoder_config["filters"][1],
        #     kernel_size=decoder_config["kernel_size"][0],
        #     act_fun=decoder_config["up_act_func"][1],
        #     padding=decoder_config["padding"][0],
        #     output_padding=decoder_config["output_padding"][0],
        #     stride=decoder_config["stride"][0]
        # )

        # self.conv1 = conv_couplet(
        #     in_channels=decoder_config["filters"][1],
        #     out_channels=decoder_config["filters"][1],
        #     kernel_size=decoder_config["kernel_size"][0],
        #     act_fun=decoder_config["act_func"][0],
        #     padding="same",
        #     stride=1
        # )

        # self.upconv2 = upconv_couplet(
        #     in_channels=decoder_config["filters"][1],
        #     out_channels=decoder_config["filters"][2],
        #     kernel_size=decoder_config["kernel_size"][1],
        #     act_fun=decoder_config["up_act_func"][1],
        #     padding=decoder_config["padding"][1],
        #     output_padding=decoder_config["output_padding"][1],
        #     stride=decoder_config["stride"][1]
        # )

        # self.conv2 = conv_couplet(
        #         in_channels=decoder_config["filters"][2],
        #         out_channels=decoder_config["filters"][2],
        #         kernel_size=decoder_config["kernel_size"][-1],
        #         act_fun=decoder_config["act_func"][-1],
        #         padding="same",
        #         stride=1
        #     )

    def forward(self, images):
        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        # print(x.shape)
        
        B, N, D = x[:, 1:, :].shape # skip cls token
        H = int(self.img_size[0]/self.patch_size[0])
        W = int(self.img_size[1]/self.patch_size[1])
        
        x = x[:, 1:, :].permute(0, 2, 1)  # [B, D, N]
        x = x.view(B, D, H, W)
        # print(x.shape)

        ## creating image from cls token - not spatial
        ## 16(batch)x64(D_MODEL)
        # x = x[:,0] # grab cls token (or do average pooling over all the patches)
        # print(x.shape)
        # add dense layer 18x36 = 648
        # x = self.dense1(x)  
        # 16, 1, 18, 36
        # print(x.shape)

        # Convolve up to 180x360
        x = self.upsamp1(x)
        # print(x.shape)
        out = self.upsamp2(x)
        # print(out.shape)
        # x = self.upconv1(x)
        # x = self.conv1(x)
        # print(x.shape)
        # x = self.upconv2(x)
        # out = self.conv2(x)
        # print(x.shape)

        mu = out[:, 0]
        sigma = F.softplus(out[:, 1]) + 1e-6
        
        return mu, sigma
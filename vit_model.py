

import torch
import torchvision
import numpy as np
import torch.nn as nn
from torchvision.transforms import Compose,ToTensor,Resize
from torch import optim
from torch.hub import tqdm

!pip install transformers sentencepiece accelerate

class ViTConfig:
  patch_size=16
  n_channels=3
  latent_size=768
  num_heads=12
  num_encoders=12
  dropout=0.1
  num_classes=16
  no_cuda=True
  batch_size=1

class PatchExtractor(nn.Module):
  def __init__(self,patch_size=16):
    super().__init__()
    self.patch_size=patch_size

  def forward(self,input_data):
    batch_size,channels,height,width = input_data.size()
    assert height % self.patch_size == 0 and width % self.patch_size == 0, \
    f"Input height ({height}) and width ({width}) must be divisible by patch_size ({self.patch_size})"

    num_patches_h = height // self.patch_size
    num_patches_w = width // self.patch_size
    num_patches = num_patches_h * num_patches_w

    patches = input_data.unfold(2,self.patch_size,self.patch_size). \
              unfold(3,self.patch_size,self.patch_size). \
              permute(0,2,3,1,4,5). \
              contiguous(). \
              view(batch_size,num_patches,-1)

    return patches

class InputEmbedding(nn.Module):
  def __init__(self,config):
    super(InputEmbedding,self).__init__()

    self.patch_size = config.patch_size
    self.n_channels = config.n_channels
    self.latent_size = config.latent_size

    use_cuda = not config.no_cuda and torch.cuda.is_available()

    self.device = torch.device("cuda" if use_cuda else "cpu")
    self.batch_size = config.batch_size
    self.input_size = self.patch_size * self.patch_size * self.n_channels

    self.linearProjection = nn.Linear(self.input_size,self.latent_size)

    self.class_token = nn.Parameter(torch.randn(self.batch_size,1,self.latent_size)).to(self.device)
    self.pos_embedding=nn.Parameter(torch.randn(self.batch_size,1,self.latent_size)).to(self.device)


  def forward(self,input_data):

    input_data=input_data.to(self.device)

    patchify=PatchExtractor(patch_size=self.patch_size)

    patches=patchify(input_data)

    linear_projection=self.linearProjection(patches).to(self.device)
    b,n,_ = linear_projection.shape
    linear_projection = torch.cat((self.class_token,linear_projection),dim=1)
    pos_embed = self.pos_embedding[:,:n+1,:]
    linear_projection += pos_embed

    return linear_projection

class EncoderBlock(nn.Module):
  def __init__(self,config):
    super(EncoderBlock,self).__init__()

    self.latent_size = config.latent_size
    self.num_heads = config.num_heads
    self.dropout = config.dropout
    self.norm = nn.LayerNorm(self.latent_size)
    self.attention = nn.MultiheadAttention(self.latent_size,self.num_heads,dropout=self.dropout)

    self.enc_MLP=nn.Sequential(
        nn.Linear(self.latent_size,self.latent_size * 4),
        nn.GELU(),
        nn.Dropout(self.dropout),
        nn.Linear(self.latent_size * 4,self.latent_size),
        nn.Dropout(self.dropout)
    )

  def forward(self,emb_patches):
    first_norm=self.norm(emb_patches)
    attention_out = self.attention(first_norm,first_norm,first_norm)[0]
    first_added = attention_out + emb_patches
    second_norm = self.norm(first_added)
    mlp_out=self.enc_MLP(second_norm)
    output = mlp_out + first_added

    return output

class ViT(nn.Module):
  def __init__(self,config:ViTConfig):
    super(ViT,self).__init__()

    # Tokenizer and Encoder Model
    from transformers import AutoTokenizer, T5EncoderModel
    self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    self.t5model = T5EncoderModel.from_pretrained("google/flan-t5-base")
    for param in self.t5model.parameters():
      param.requires_grad=False


    self.num_encoders = config.num_encoders
    self.latent_size = config.latent_size
    self.num_classes = config.num_classes

    self.dropout = config.dropout

    self.embedding = InputEmbedding(config)

    self.encoders = nn.ModuleList([EncoderBlock(config) for _ in range(self.num_encoders)])
    self.ConvHead=nn.Sequential(
        nn.Conv2d(in_channels=36,out_channels=18,kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=18,out_channels=9,kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=9,out_channels=3,kernel_size=1),
        nn.ReLU()
    )

  def embedder(self,im,prp):
    prp_out=self.t5model(input_ids=self.tokenizer(prp, return_tensors="pt").input_ids)
    prp_out=prp_out.last_hidden_state
    im_out=self.embedding(im)
    return torch.cat((im_out,prp_out),1)

  def forward(self,x):

    # outputs = self.t5model(input_ids=self.tokenizer(prompt, return_tensors="pt").input_ids)
    # last_hidden_states = outputs.last_hidden_state

    # enc_output = self.embedding(x)

    # enc_output=torch.cat((enc_output,last_hidden_states),1)
    enc_output=x

    list_out=[]
    for enc_layer in self.encoders:
      enc_output = enc_layer(enc_output)
      list_out.append(enc_output)

    list_out=torch.cat((list_out))

    conv_in=list_out.unsqueeze_(0).permute((0,1,3,2)).unfold(3,1024,1024).reshape((1,-1,1024,1024))

    #print(list_out.shape)

    return self.ConvHead(conv_in)

vit_config=ViTConfig()
model=ViT(vit_config)

model(model.embedder(torch.randn((1,3,1024,1024)),"Hello How are you?")).shape

# import torch.nn as nn
# import torch
# import torch.nn.functional as F

# # Define the input tensor with shape [batch_size, in_channels, height, width]
# input_tensor = torch.randn(1, 12, 4103, 768)

# def ins(img):


#   t1=img.permute((0,1,3,2)).unfold(3,1024,1024).reshape((1,-1,1024,1024))
#   print(t1.shape)
#   # t1=t1.unfold(3,1024,1024)
#   # print(t1.shape)
#   # t1=t1.reshape((1,-1,1024,1024))
#   # print(t1.shape)

#   conv1=nn.Conv2d(in_channels=36,out_channels=18,kernel_size=1)

#   conv2=nn.Conv2d(in_channels=18,out_channels=9,kernel_size=1)
#   conv3=nn.Conv2d(in_channels=9,out_channels=3,kernel_size=1)

#   out=conv3(conv2(conv1(t1)))
#   return out.shape

# ins(input_tensor)

# input_tensor = torch.randn(12, 4103, 768)
# print(input_tensor.shape)
# input_tensor.unsqueeze_(0).shape
# print(input_tensor.shape)


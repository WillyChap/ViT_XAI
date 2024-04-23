import torch
from Simple_ViT import SimpleViT

if __name__ == "__main__":
    v = SimpleViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 4,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048
    )
    
    img = torch.randn(1, 3, 256, 256)  #Batch, Variable, Lon, Lat
    preds = v(img) # (1, num_classes)
    print('I did it.')
    
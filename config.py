import torch 

epochs = 150
batch_size = 1
lr = 0.001
num_classes = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
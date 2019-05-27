import os
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import time
from time import gmtime, strftime
from torch.utils.data.dataset import Dataset
from random import randint

BS = 1000 

""" Load data from the mnist dataset """
def load_data(mini=False):
    fname_img_tr = os.path.join(".", "data/MNIST/raw/train-images-idx3-ubyte")
    fname_lbl_tr = os.path.join(".", "data/MNIST/raw/train-labels-idx1-ubyte")
    fname_img_te = os.path.join(".", "data/MNIST/raw/t10k-images-idx3-ubyte")
    fname_lbl_te = os.path.join(".", "data/MNIST/raw/t10k-labels-idx1-ubyte")

    preload =   os.path.exists(os.path.join(".", "data/lin_train.pt")) and \
                os.path.exists(os.path.join(".", "data/lin_test.pt")) if not mini else\
                os.path.exists(os.path.join(".", "data/lin_train_mini.pt")) and \
                os.path.exists(os.path.join(".", "data/lin_train_mini.pt"))

    if(preload and not mini):
        print("Pre-processed files found, loading.")
        tr_img, tr_lbls = torch.load(os.path.join(".", "data/lin_train.pt"))
        te_img, te_lbls = torch.load(os.path.join(".", "data/lin_test.pt"))
        return tr_img, tr_lbls, te_img, te_lbls
    elif(preload and mini):
        print("Pre-processed files found, loading.")
        tr_img, tr_lbls = torch.load(os.path.join(".", "data/lin_train_mini.pt"))
        te_img, te_lbls = torch.load(os.path.join(".", "data/lin_test_mini.pt"))
        return tr_img, tr_lbls, te_img, te_lbls
    else:
        print("No preprocessed local data. Processing and saving.")


    """ Using ordinary lists temporarily before converting this list of data into np -> pytorch array.
    Unsure if there's a better way """
    imgs, lbls = [], []

    flbl = open(fname_lbl_tr, 'rb')
    fimg = open(fname_img_tr, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))

    amount = 2000 if mini else size

    for i in range(amount):
        img = struct.unpack("B"*rows*cols, fimg.read(rows*cols))
        lbl = struct.unpack("B", flbl.read(1))
        imgs.append(img)
        lbls.append(lbl[0])

    fimg.close()
    flbl.close()

    tr_imgs = torch.from_numpy(np.array(imgs)).to(dtype= torch.float)
    tr_lbls = torch.from_numpy(np.array(lbls)).to(dtype= torch.float)

    if mini:
        imgs, lbls = [], []
        flbl = open(fname_lbl_te, 'rb')
        fimg = open(fname_img_te, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))

        for i in range(int(amount/2)):
            img = struct.unpack("B"*rows*cols, fimg.read(rows*cols))
            lbl = struct.unpack("B", flbl.read(1))
            imgs.append(img)
            lbls.append(lbl[0])

        fimg.close()
        flbl.close()

        te_imgs = torch.from_numpy(np.array(imgs)).to(dtype=torch.float)
        te_lbls = torch.from_numpy(np.array(lbls)).to(dtype=torch.float)

        with open(os.path.join(".", "data/lin_train_mini.pt"), "wb") as f:
            torch.save((tr_imgs, tr_lbls), f)
        
        with open(os.path.join(".", "data/lin_test_mini.pt"), "wb") as f:
            torch.save((te_imgs, te_lbls), f)

        return tr_imgs, tr_lbls, te_imgs, te_lbls
    else:
        imgs, lbls = [], []
        flbl = open(fname_lbl_te, 'rb')
        fimg = open(fname_img_te, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))

        for i in range(size):
            img = struct.unpack("B"*rows*cols, fimg.read(rows*cols))
            lbl = struct.unpack("B", flbl.read(1))
            imgs.append(img)
            lbls.append(lbl[0])

        fimg.close()
        flbl.close()

        te_imgs = torch.from_numpy(np.array(imgs)).to(dtype=torch.float)
        te_lbls = torch.from_numpy(np.array(lbls)).to(dtype=torch.float)

        with open(os.path.join(".", "data/lin_train.pt"), "wb") as f:
            torch.save((tr_imgs, tr_lbls), f)
        
        with open(os.path.join(".", "data/lin_test.pt"), "wb") as f:
            torch.save((te_imgs, te_lbls), f)

        return tr_imgs, tr_lbls, te_imgs, te_lbls

class CustomMnistDataset(Dataset):
    def __init__(self, data):
        self.imgs = data[0]
        self.lbls = data[1]
        
    def __getitem__(self, index):
        img = self.imgs[index]
        lbl = self.lbls[index]
        #One hot encoding the labels when retrieved since I don't know how to do it in a vectorized manner later on.
        lbl_oh = to_onehot(lbl)

        return img, lbl_oh 

    def __len__(self):
        return len(self.imgs)

""" Function to doublecheck that the labels matches the images. Prints a random digit/label in the python console. """
def print_random_sample(imgs, lbls):
    r_idx = randint(0, len(imgs))
    img, lbl = imgs[r_idx], lbls[r_idx]
    print(f"Predicted number: {lbl.item()}")
    tmp_str = ""
    for i in range(len(img)):
        s = "x" if img[i] > 0 else " "
        tmp_str += s
        if i % 28 == 0:
            tmp_str += "\n"

    print(tmp_str)

""" Manual one hot encoding function. Unsure if library has one."""
def to_onehot(num):
    l = torch.zeros(10, dtype=torch.float)
    l[int(num)] = 1
    return l

class Lin_Mnist(nn.Module):
    def __init__(self, am_l, am_n):
        super(Lin_Mnist, self).__init__()
        self.inp = nn.Linear(784, am_n)  # Input image, hidden layer
        self.h_layers = nn.ModuleList([nn.Linear(am_n, 10 if i == am_l-1 else am_n) for i in range(am_l)])

    def forward(self, x):
        #Layer structure set up above, when passing through last layer size should be 10.
        x = self.inp(x)
        for layer in self.h_layers:
           x = layer(x) 
        return F.softmax(x, dim=1)

""" Training phase."""
def train(model, opt, crt, loader):
    model.train()
    for b_id, (b_x, b_y) in enumerate(loader):
       b_x, b_y = b_x.to(device), b_y.to(device) 
       #Reset the nablas.
       opt.zero_grad()
       #Get predictions for the entire batch
       y_hat = model(b_x) 
       #Calculate individual loss for every prediction
       loss = crt(y_hat, b_y)
       #Update the nablas/backpropagate
       loss.backward()
       #Perform the descent step
       opt.step()

""" Testing phase"""
def test(model, crt, loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for b_x, b_y in loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            #Get predictions
            y_hat = model(b_x)
            #Do argmax on both to ensure the same format. Actual is one-hot encoded, but this was the easiest way to convert that i could figure out.
            pred = y_hat.argmax(dim=1, keepdim=True)
            actual = b_y.argmax(dim=1, keepdim=True)
            #Some sort of vectorized magic I don't fully understand, but it works. 
            correct += pred.eq(actual.view_as(pred)).sum().item()

    return correct/len(loader.dataset)

""" Function to quickly loop through and try different parameters to find an optimal setting. """
def test_setting(tr_loader, te_loader, am_l, am_n, lr, epoch):
    criterion = nn.MSELoss(reduction='sum')
    model = Lin_Mnist(am_l, am_n).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.1)
    accuracy = 0
    for i in range(epoch):
        train(model, optimizer, criterion, tr_loader)
        accuracy = test(model, criterion, te_loader, i) 
    print(f'layer_am: {am_l}\tneuron_am: {am_n}\tlr: {lr}\tepochs: {epoch}\taccuracy: {accuracy*100}%')
    return accuracy

def main():
    #Get training and test data. Goes fast once preprocessed and saved locally 
    tr_imgs, tr_lbls, te_imgs, te_lbls = load_data(mini=False)

    tr_data = [tr_imgs, tr_lbls]
    te_data = [te_imgs, te_lbls]

    #Create custom dataset, add into loader
    te_ds = CustomMnistDataset(te_data)
    tr_ds = CustomMnistDataset(tr_data)
    te_dl = torch.utils.data.DataLoader(dataset=te_ds, batch_size=BS, shuffle=True)
    tr_dl = torch.utils.data.DataLoader(dataset=tr_ds, batch_size=BS, shuffle=True)

    b_acc = 0
    best = {}

    res_fname = "results/test-{}.csv".format(strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
    res_f = open(res_fname, "w+")
    res_f.write("amount_layers,amount_neurons,learning_rate,accuracy(%),time_elapsed\n")
    res_f.close()


    #layer_am: 6     neuron_am: 83   lr: 1.5e-05     epochs: 3       accuracy: 91.86999999999999

    for lr in range(1, 10):
        lr = 1 if lr == 0 else lr
        for am_l in range(1, 2):
            for am_n in range(8, 9):
                t_acc = 0
                lr_mod = (lr * 1e-07) + 1.80e-05
                t0 = time.time()
                for i in range(5):
                    t_acc += test_setting(tr_dl, te_dl, am_l, am_n, lr_mod, 10)
                t1 = time.time()
                t_acc /=3

                res_f = open(res_fname, "a")
                res_f.write("{},{},{},{:.1f},{:.2f}\n".format(am_l, am_n, lr_mod, t_acc*100, t1-t0))
                res_f.close()

                if t_acc > b_acc:
                    b_acc = t_acc
                    best = {
                        "am_l": am_l,
                        "am_n": am_n,
                        "lr": lr,
                        "acc": (t_acc * 100),
                        "t": t1-t0
                    }
    
    res_f = open(res_fname, "a")
    res_f.write("{},{},{},{},{}\n".format(best["am_l"], best["am_n"], best["lr"], best["acc"], best["t"]))
    res_f.close()
    
#Global declaration of device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#Needed to make the workers actually work. They don't work very well though. Faster to leave them off.
if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.utils.data


from matplotlib import pyplot as plt

## CUSTOM
from dataloader.utils import load_mnist
from models.CNN import CNN

def get_loader(X, y, batch_size=64):
    train = torch.utils.data.TensorDataset(torch.from_numpy(X).float(),
                                       torch.from_numpy(y).long())
    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size=batch_size)
    return train_loader


def train_epoch(model, optimizer, train_loader, criterion, device):
    """
    for each batch
    performs forward and backward pass and parameters update

    Input:
    model: instance of model (example defined above)
    optimizer: instance of optimizer (defined above)
    train_loader: instance of DataLoader

    Returns:
    nothing

    Do not forget to set net to train mode!
    """
    ### your code here
    model.train()
    for it, traindata in enumerate(train_loader):
        train_inputs, train_labels = traindata
        train_inputs = train_inputs.to(device) 
        train_labels = train_labels.to(device)
        train_labels = torch.squeeze(train_labels)

        model.zero_grad()        
        # train_inputs = train_inputs.permute(0,1,2,3)
        # print(train_inputs.shape)
        # print(train_inputs.t())
        output = model(train_inputs)
        # output = model(train_inputs.t()) # pay attention here!
        # print(train_labels.shape)
        loss = criterion(output, train_labels.long())
        loss.backward()
        optimizer.step()


def evaluate_loss_acc(loader, model, criterion, device):
    """
    Evaluates loss and accuracy on the whole dataset

    Input:
    loader:  instance of DataLoader
    model: instance of model (examle defined above)

    Returns:
    (loss, accuracy)

    Do not forget to set net to eval mode!
    """
    ### your code here
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    for it, data in enumerate(loader):
        inputs, labels = data
        inputs = inputs.to(device) 
        labels = labels.to(device)
        labels = torch.squeeze(labels)

        output = model(inputs) # pay attention here!
        loss = criterion(output, labels.long())# + torch.norm(WW^T - I)
        total_loss += loss.item()
        # print(labels)
        # calc testing acc        
        # pred = output.view(-1) > 0.5
        pred = output.argmax(dim=1)
        correct = pred == labels.byte()
        total_acc += torch.sum(correct).item() / len(correct)

    total = it + 1
    return total_loss / total, total_acc / total

def train(model, opt, train_loader, test_loader, criterion, n_epochs, \
        device, verbose=True):
    """
    Performs training of the model and prints progress

    Input:
    model: instance of model (example defined above)
    opt: instance of optimizer
    train_loader: instance of DataLoader
    test_loader: instance of DataLoader (for evaluation)
    n_epochs: int

    Returns:
    4 lists: train_log, train_acc_log, val_log, val_acc_log
    with corresponding metrics per epoch
    """
    train_log, train_acc_log = [], []
    val_log, val_acc_log = [], []

    for epoch in range(n_epochs):
        train_epoch(model, opt, train_loader, criterion, device)
        train_loss, train_acc = evaluate_loss_acc(train_loader,
                                                model, criterion,
                                                device)
        val_loss, val_acc = evaluate_loss_acc(test_loader, model,
                                            criterion, device)

        train_log.append(train_loss)
        train_acc_log.append(train_acc)

        val_log.append(val_loss)
        val_acc_log.append(val_acc)

        if verbose:
            print (('Epoch [%d/%d], Loss (train/test): %.4f/%.4f,'+\
            ' Acc (train/test): %.4f/%.4f' )
                %(epoch+1, n_epochs, \
                    train_loss, val_loss, train_acc, val_acc))

    return train_log, train_acc_log, val_log, val_acc_log

        
if __name__ == '__main__' :
    X_train, y_train, X_test, y_test = load_mnist()

    # shuffle data
    np.random.seed(0)
    idxs = np.random.permutation(np.arange(X_train.shape[0]))
    X_train, y_train = X_train[idxs], y_train[idxs]

    X_train.shape

    cnn = CNN( )

    criterion = nn.CrossEntropyLoss() # loss includes softmax

    device = torch.device('cpu')
    cnn = cnn.to(device)

    train_loader = get_loader(X_train[:15000], y_train[:15000])
    test_loader = get_loader(X_test, y_test)
    
    learning_rate = 0.01
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    num_epochs = 5
    a, b, c, d = train(train_loader=train_loader, test_loader=test_loader, model=cnn, criterion=criterion,opt=optimizer, device=device, n_epochs=num_epochs)
    # print( a ,b ,c ,d)

    save_path = './models/CNN.pth'
    torch.save(cnn.state_dict(), save_path)

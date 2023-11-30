from tqdm import tqdm
import torch.nn as nn
import numpy as np
import time
import torch
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EarlyStopping:
    def __init__(self, patience = 20, stop_epoch = 50, verbose=False):

        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def Train_model(model, trainLoaders, args, valLoaders = [], criterion = None, optimizer = None, fold = False):
    
    since = time.time()    
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []  
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = args.patience, stop_epoch = args.minEpochToTrain, verbose = True)    
    for epoch in range(args.max_epochs):
        phase = 'train'
        print('Epoch {}/{}\n'.format(epoch, args.max_epochs - 1))
        print('\nTRAINING...\n')        
        model.train() 
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels, inputs2, labels2 in tqdm(trainLoaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs2 = inputs2.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs, inputs2)
                loss = criterion(outputs, labels)
                _, y_hat = torch.max(outputs, 1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(y_hat == labels.data)
        epoch_loss = running_loss / len(trainLoaders.dataset)
        epoch_acc = running_corrects.double() / len(trainLoaders.dataset)        
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)        
        print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print()        
        if valLoaders:
            print('VALIDATION...\n')
            phase = 'val'
            model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels, inputs2, labels2 in tqdm(valLoaders):
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs2 = inputs2.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, inputs2)
                    loss = criterion(outputs, labels)
                    _, y_hat = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(y_hat == labels.data)
            val_loss = running_loss / len(valLoaders.dataset)
            val_acc = running_corrects.double() / len(valLoaders.dataset)
            val_acc_history.append(val_acc)
            val_loss_history.append(val_loss)
            print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, val_loss, val_acc))
            if fold == 'FULL':
                ckpt_name = os.path.join(args.result_dir, "bestModel")
            else:
                ckpt_name = os.path.join(args.result_dir, "bestModelFold" + fold)
            early_stopping(epoch, val_loss, model, ckpt_name = ckpt_name)
            if early_stopping.early_stop:
                print('-' * 30)
                print("The Validation Loss Didn't Decrease, Early Stopping!!")
                print('-' * 30)
                break
            print('-' * 30)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, train_loss_history, train_acc_history, val_acc_history, val_loss_history 

    
def Valid_model(model, dataloaders):
    
    phase = 'test'
    model.eval()
    probsList = []

    for inputs, labels, inputs2, labels2 in tqdm(dataloaders):
        inputs = inputs.to(device)
        inputs2 = inputs2.to(device)
        with torch.set_grad_enabled(phase == 'train'):            
            probs = nn.Softmax(dim=1)(model(inputs, inputs2))
            probsList = probsList + probs.tolist()
    return probsList 

    
    
    
    
    
    
    
    
    
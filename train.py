# train.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import TaxiDriverClassifier
from extract_feature import load_data, preprocess_data
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
import numpy as np
from logger import Logger
import time
from torch.optim import lr_scheduler

torch.manual_seed(42)

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print(device)
else:
  device = torch.device("cpu")
  print(device)

## Using np.load to save time taken to load the data and preprocess it. Please load the data as per your method over here. 
X_combined = np.load(r"path\to\X_combined.npy")
y_combined = np.load(r"path\to\y_combined.npy")


class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {'X': self.X[idx], 'y': self.y[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class ToTensor(object):
    def __call__(self, sample):
        
        X, y = sample['X'], sample['y']
        return {'X': torch.tensor(X, dtype=torch.float32),
                'y': y} 

    
class TaxiDriverDataset(Dataset):
    """
    Custom dataset class for Taxi Driver Classification.
    Handles loading and preparing data for the model
    """
    def __init__(self, X, y, device):
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_val, self.y_val, test_size=0.2, random_state=42)
        self.transform = ToTensor()
        self.train_dataset = CustomDataset(self.X_train, self.y_train, transform=self.transform)
        self.val_dataset = CustomDataset(self.X_val, self.y_val, transform=self.transform)
        self.test_dataset = CustomDataset(self.X_test, self.y_test, transform=self.transform)
        self.batch_size =  64
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        ###########################

    def train(self, model, optimizer, criterion, train_loader, device):
        """
        Function to handle the training of the model.
        Iterates over the training dataset and updates model parameters.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        model = model.to(device)
        model.train()  
        train_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, sample in enumerate(train_loader):
            # print(batch_idx+1)
            inputs, labels = sample['X'], sample['y']
            inputs, labels = inputs.to(device), labels.to(device)
            # labels = F.one_hot(labels, num_classes=5)            

            optimizer.zero_grad()  
            outputs = model(inputs)
            outputs = torch.squeeze(outputs,0)

            loss = criterion(outputs, labels) 
            loss.backward()  
            optimizer.step() 

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = correct_predictions / total_samples

        ###########################
        return train_loss, train_acc

    #Define the testing function
    def evaluate(self, model, criterion, test_loader, device):
        """
        Function to evaluate the model performance on the validation set.
        Computes loss and accuracy without updating model parameters.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        model.eval()
        test_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for sample in test_loader:
                
                inputs, labels = sample["X"], sample["y"] 
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                outputs = torch.squeeze(outputs, 0)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs,1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        test_loss /= len(test_loader)
        test_acc = correct_predictions / total_samples
        model.train()
        ###########################
        return test_loss, test_acc

    def train_model(self, train_loader, val_loader):
        """
        Main function to initiate the model training process.
        Includes loading data, setting up the model, optimizer, and criterion,
        and executing the training and validation loops.
        """

        ###########################
        # YOUR IMPLEMENTATION HERE #
        learning_rate = 0.001
        epochs = 75
        input_dim, hidden_dim, output_dim = 8, 32, 5
        logger.log(tag='args', learning_rate=learning_rate, epochs=epochs, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        model = TaxiDriverClassifier(input_dim, hidden_dim, output_dim, num_layers=4, dropout_rate=0.2)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        _best_loss = 0.0
        _best_acc = 0.0
        for i in range(epochs):
            _start = time.time()
            print(f"{i+1}/{epochs} epochs done")
            train_loss, train_acc = taxi_dataset.train(model, optimizer, criterion, train_loader, device)
            print("Train Loss = ", train_loss, "Train Accuracy = ", train_acc)
            logger.log(tag='train', epoch=i+1, loss=train_loss, acc=train_acc, time=time.time()-_start)
            valid_loss, valid_acc = taxi_dataset.evaluate(model, criterion, val_loader, device)
            print("Validation Loss = ", valid_loss, "Validation Accuracy = ", valid_acc)
            logger.log(tag='val', epoch=i+1, loss=valid_loss, acc=valid_acc, time=time.time()-_start)

            if valid_loss < _best_loss:
                _best_loss = valid_loss
                logger.log(tag='model_loss', loss=_best_loss)
                torch.save(model.state_dict(), 'logs/trial_logs9/model/best_loss_model.pt')

            if valid_acc > _best_acc:
                _best_acc = valid_acc
                logger.log(tag='model_acc', acc=_best_acc)
                torch.save(model.state_dict(), 'logs/trial_logs9/model/best_acc_model.pt')

            logger.log(tag='plot')
        ###########################


if __name__ == '__main__':

    global logger

    logger = Logger(log_dir='logs/trial_logs9')

    taxi_dataset = TaxiDriverDataset(X_combined, y_combined, device)
    taxi_dataset.train_model(taxi_dataset.train_loader, taxi_dataset.val_loader)
import os
import time
import matplotlib.pyplot as plt

class Logger:
    """ Class to log the training process

    Args:
        log_dir (str): path to the log directory

    Attributes:
        log_dir (str): path to the log directory
        plot_dir (str): path to the plot directory
        media_dir (str): path to the media directory
        log_file_path (str): path to the log file
        log_file (file): log file
        train_loss (list): list of training data
        val_loss (list): list of validation data

    Methods:
        log(tag, **kwargs): log the data
        plot(data, name, path): plot the data
        plot_both(data1, data2, name, path): plot the data
        draw(epoch, img): draw the image
    """

    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.plot_dir = os.path.join(log_dir, 'plots')
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)

        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.log_file_path = self.log_dir + '/logs.txt'
        self.log_file = open(self.log_file_path, 'w')
        self.log_file.write('Logs date and time: '+time.strftime("%d-%m-%Y %H:%M:%S")+'\n\n')

        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def log(self, tag, **kwargs):
        """ Log the data

        Args:
            tag (str): tag for the data
            **kwargs: data
        """

        self.log_file = open(self.log_file_path, 'a')

        if tag == 'args':
            self.log_file.write('Training Args:\n')
            for k, v in kwargs.items():
                self.log_file.write(str(k)+': '+str(v)+'\n')
            self.log_file.write('#########################################################\n\n')
            self.log_file.write(f'Starting Training... \n')

        elif tag == 'train':
            self.train_loss.append([kwargs['loss']])
            self.train_acc.append([kwargs['acc']])
            self.log_file.write(f'Epoch: {kwargs["epoch"]} \t Train Loss: {kwargs["loss"]} \t Train Acc: {kwargs["acc"]} \t Avg Time: {kwargs["time"]} secs\n')

        elif tag == 'val':
            self.val_loss.append([kwargs['loss']])
            self.val_acc.append([kwargs['acc']])
            self.log_file.write(f'Epoch: {kwargs["epoch"]} \t Val Loss: {kwargs["loss"]} \t Val Acc: {kwargs["acc"]} \t Avg Time: {kwargs["time"]} secs\n')

        elif tag == 'model_loss':
            self.log_file.write('#########################################################\n')
            self.log_file.write(f'Saving best model... Val Loss: {kwargs["loss"]}\n')
            self.log_file.write('#########################################################\n')

        elif tag == 'model_acc':
            self.log_file.write('#########################################################\n')
            self.log_file.write(f'Saving best model... Val Accuracy: {kwargs["acc"]}\n')
            self.log_file.write('#########################################################\n')

        elif tag == 'plot':
            self.plot(self.train_loss, name='Train Loss', path=self.plot_dir)
            self.plot(self.train_acc, name='Train Accuracy', path=self.plot_dir)
            self.plot(self.val_loss, name='Val Loss', path=self.plot_dir)
            self.plot(self.val_acc, name='Val Accuracy', path=self.plot_dir)
            self.plot_both(self.train_loss, self.val_loss, name='Loss', path=self.plot_dir)
            self.plot_both(self.train_acc, self.val_acc, name='Accuracy', path=self.plot_dir)

        self.log_file.close()

    def plot(self, data, name, path):
        """ Plot the data

        Args:
            data (list): data
            name (str): name of the data
            path (str): path to the plot
        """

        plt.plot(data)
        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.title(name+' vs. Epochs')
        plt.savefig(os.path.join(path, name+'.png'), dpi=600 ,bbox_inches='tight')
        plt.close()

    def plot_both(self, data1, data2, name, path):
        """ Plot data1 and data2 in the same plot

        Args:
            data1 (list): data1
            data2 (list): data2
            name (str): name of the data
            path (str): path to the plot
        """

        plt.plot(data1, label='Train')
        plt.plot(data2, label='Val')
        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.title(name+' vs. Epochs')
        plt.legend()
        plt.savefig(os.path.join(path, name+'.png'), dpi=600 ,bbox_inches='tight')
        plt.close()
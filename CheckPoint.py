import os
import torch

def generate_unique_logpath(logdir:str, raw_run_name:str):
    i = 0
    while(True):
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

class ModelCheckpoint:

    def __init__(self, filepath:str, model:torch.nn.Module):
        self.min_loss = None
        self.filepath = filepath
        self.model = model

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            #torch.save(self.model.state_dict(), self.filepath)
            torch.save(self.model, self.filepath)
            self.min_loss = loss

def save_path():
    top_logdir = "./logs"
    if not os.path.exists(top_logdir):
        os.mkdir(top_logdir)

    logdir = generate_unique_logpath(top_logdir,"Experience")
    print("Logging to {}".format(logdir))

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    return logdir

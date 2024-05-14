import torch

class EarlyStopping:
    """Early stops the training if training error doesn't improve after a given patience and saves the best model."""
    def __init__(self, patience=7, verbose=False, delta=0, path='best_model.pth'):
        """
        Args:
            patience (int): How long to wait after the last time training error improved.
            verbose (bool): If True, prints a message for each training error improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the best model file.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_error, model):
        score = -val_error
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_error, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if score > self.best_score:  # Check for improvement
                self.best_score = score
                self.save_checkpoint(val_error, model)
                self.counter = 0  # Reset the counter on improvement

    def save_checkpoint(self, val_error, model):
        '''Saves model when training error decreases.'''
        if self.verbose:
            print(f'Val error decreased to {val_error:.4f}°. Saving model...\n')
        torch.save(model.state_dict(), self.path)
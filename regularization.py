from config import config
l2 = config["training"]["l2_scheduler"]

# class RegularizationScheduler:
#     def __init__(self, optimizer, 
#                  init_weight_decay=l2["init"], 
#                  increase_factor=l2["increase_factor"], 
#                  decrease_factor=l2["decrease_factor"], 
#                  threshold=l2["threshold"], 
#                  patience=l2["patience"],
#                  verbose=l2["verbose"]):
#         self.optimizer = optimizer
#         self.weight_decay = init_weight_decay
#         self.increase_factor = increase_factor  # Factor by which to increase weight decay
#         self.decrease_factor = decrease_factor  # Factor by which to decrease weight decay
#         self.threshold = threshold  # Threshold for deciding significant overfitting
#         self.patience = patience  # How many epochs to wait before increasing weight decay
#         self.verbose = verbose
#         self.epoch = 0
#         self.best_val_loss = float('inf')
#         self.worse_epochs = 0  # Counter for epochs without improvement

#     def step(self, train_loss, val_loss):
#         """Update the weight decay based on training and validation loss."""
#         if val_loss < self.best_val_loss:
#             self.best_val_loss = val_loss
#             self.worse_epochs = 0
#         else:
#             self.worse_epochs += 1

#         if self.worse_epochs >= self.patience:
#             if (train_loss < val_loss * self.threshold):
#                 # Increase regularization to combat overfitting
#                 new_weight_decay = self.weight_decay * self.increase_factor
#                 for param_group in self.optimizer.param_groups:
#                     param_group['weight_decay'] = new_weight_decay
#                 self.weight_decay = new_weight_decay
#                 if self.verbose:
#                     print(f"Increasing weight decay to {new_weight_decay}")
#             self.worse_epochs = 0
#         self.epoch += 1
        

class RegularizationScheduler:
    def __init__(self, optimizer, 
                 init_weight_decay=l2["init"], 
                 increase_factor=l2["increase_factor"], 
                 decrease_factor=l2["decrease_factor"], 
                 threshold=l2["threshold"], 
                 patience=l2["patience"],
                 verbose=l2["verbose"]):
        self.optimizer = optimizer
        self.weight_decay = init_weight_decay
        self.increase_factor = increase_factor  # Factor by which to increase weight decay
        self.decrease_factor = decrease_factor  # Factor by which to decrease weight decay
        self.threshold = threshold  # Threshold for deciding significant overfitting
        self.patience = patience  # How many epochs to wait before increasing weight decay
        self.verbose = verbose
        self.epoch = 0
        self.best_val_median = float('inf')
        self.worse_epochs = 0  # Counter for epochs without improvement

    def step(self, train_median, val_median):
        """Update the weight decay based on training and validation median errors."""
        if val_median < self.best_val_median:
            self.best_val_median = val_median
            self.worse_epochs = 0
        else:
            self.worse_epochs += 1

        if self.worse_epochs >= self.patience:
            if (train_median < val_median * self.threshold):
                # Increase regularization to combat overfitting
                new_weight_decay = self.weight_decay * self.increase_factor
                for param_group in self.optimizer.param_groups:
                    param_group['weight_decay'] = new_weight_decay
                self.weight_decay = new_weight_decay
                if self.verbose:
                    print(f"Increasing weight decay to {new_weight_decay:.6f}\n")
            self.worse_epochs = 0
        self.epoch += 1
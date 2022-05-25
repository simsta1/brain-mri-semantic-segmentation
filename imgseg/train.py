import torch
import numpy as np
import pandas as pd


class TrainingInterface():
    
    def __init__(self, model, name, writer: object = None):
        """
        Training Interface Wrapper class for the training of neural network classifier
        in pytorch. Only applicable for image classification.
        
        Params:
        -------------------
        model: (torch.model)     Neural Network class Pytorch     
        name: (str)              Name of Neural Network
        writer: (object)         If true uses wandb to log all outputs during training and inference
        
        dev:                     Device Cuda or cpu
        train_losses:            Training losses recorded during training
        eval_losses:             Validation Losses recorded during training
        """
        self.model = model
        self.name = name 
        self.writer = writer
        
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.epoch = 0
        self.batch_train_loss = []
        self.batch_val_loss = []
        self.epoch_train_loss = []
        self.epch_val_loss = []
        

    def print_network(self):
        """
        Prints networks and its layers.
        """
        print(self.model)
    
    def print_total_params(self, return_=False):
        """
        Prints total params.
        
        Params:
        -------------      
        return_:            if return the the result will be returned        
        """
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
                                  
        if not return_:
            print(50 * '=')
            print(f'{self.name} | Trainable Parameters: {pytorch_total_params}')
            print(50 * '=')
        else:
            return '{}\n{}\n{}'.format(50 * '=', pytorch_total_params, 50 * '=')
        
    def train(self, criterion, optimizer, n_epochs, dataloader_train, 
              dataloader_val=None, epsilon=.0001, verbose=True):
        self.model.to(self.dev)
        criterion.to(self.dev)

        self.model.train()
        overall_length = len(dataloader_train)
        with tqdm(total=n_epochs*overall_length) as pbar:
            for epoch in range(n_epochs):  # loop over the dataset multiple times
                running_loss, val_loss = 0., 0.
                for i, data in enumerate(dataloader_train):
                    # get the inputs; data is a list of [inputs, labels]
                    images, true_mask = data
                    images, true_mask = images.to(self.dev), true_mask.to(self.dev)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    out = self.model(images)
                    loss = criterion(out, true_mask)
                    loss.backward()
                    optimizer.step()

                    # calc and print stats
                    self.batch_train_loss.append(loss.item())
                    if self.writer != None:
                        self.writer.log({'train_batch_loss': loss.item()})
                        
                    running_loss += loss.item()                
                    pbar.set_description('Epoch: {}/{} // Running Loss: {} '.format(epoch+1, n_epochs, 
                                                                                    np.round(running_loss, 3)))   
                    pbar.update(1)
                
                self.epoch_train_loss.append(running_loss)
                if self.writer != None:
                    self.writer.log({'train_epoch_loss': running_loss})

                if dataloader_val:
                    length_dataloader_val = len(dataloader_val)
                    val_loss = 0.
                    for i, data in enumerate(dataloader_val):
                        pbar.set_description(f'Epoch: {epoch+1}/{n_epochs} // Eval-Loop: {i+1}/{length_dataloader_val}')
                        self.model.eval()
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = data
                        inputs, labels = inputs.to(self.dev), labels.to(self.dev)
                        with torch.no_grad():
                            outputs = self.model(inputs)
                            eval_loss = criterion(outputs, labels)
                            val_loss += eval_loss.item()
                            self.batch_val_loss.append(eval_loss.item())
                            if self.writer != None:
                                self.writer.log({'val_batch_loss': eval_loss.item()})
                        self.model.train()  
                    self.epoch_val_loss.append(val_loss)
                    
                    if self.writer != None:
                        self.writer.log({'val_epoch_loss': val_loss})
                
                # Update epoch
                self.epoch += 1
                
                if verbose:
                    print('Epoch {}/{}: [Train-Loss = {}] || [Validation-Loss = {}]'.format(self.epoch, n_epochs, 
                                                                                         np.round(running_loss, 3),     
                                                                                         np.round(val_loss, 3)))     
                if epoch > 0:
                    if epsilon > np.abs(loss_before - running_loss):
                        print(20*'=', 'Network Converged', 20*'=')
                        break
                loss_before = running_loss
                    
        return self
    
    def segment(self, dataloader, return_images: bool = True, return_prob: bool = True, 
                disable_pbar: bool = False):
        """
        Returns true and predicted labels for prediction
        Params:
        ---------
        model:           Pytorch Neuronal Net
        dataloader:      batched Testset
        return_images:   If true returns images
        return_prob:     If true returns predicted probabilities
        disable_pbar:    If true disables pbar
        returns:
        ----------
        (y_true, y_pred, y_images, y_prob): 
            y_true       True labels
            y_pred:      Predicted Labels
            y_prob:      Predicted Probability (empty if return_prob = False)
            y_images:    Images (empty if return_images = False)
        """
        self.model.to(self.dev)
        self.model.eval()
        y_pred, y_true, y_images, y_prob = [], [], [], [] 
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Calculate Predictions', disable=disable_pbar):
                images, labels = batch
                images, labels = images.to(self.dev), labels.to(self.dev)
                y_probs = F.sigmoid(self.model(images))
                
                #y_probs = F.softmax(self.model(images), dim = -1)
                
                if return_images:
                    y_images.append(images.cpu())
                y_prob.append(y_probs.cpu()) 
                y_true.append(labels.cpu())
                
        if return_images:
            y_images = torch.cat(y_images, dim = 0)
        
        y_prob = torch.cat(y_prob, dim = 0)
        y_true = torch.cat(y_true, dim = 0)
        y_pred = torch.argmax(y_prob, 1)        

        return (y_true, 
                y_pred, 
                y_images if return_images else None,
                y_prob if return_prob else None)
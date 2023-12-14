# ==========================================
# Train and Validation Class
# ===========================================

from tqdm import tqdm

import torch


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

class EFRTraining():
    def __init__(self, model, training_loader, optimizer, epochs, device):
        self.model = model
        self.training_loader = training_loader
        self.optimizer = optimizer 
        self.EPOCHS = epochs
        self.device = device


    def train_steps(self, epoch):

        total_loss = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        n_correct = 0

        loop = tqdm(enumerate(self.training_loader, 0), total=len(self.training_loader))
        for _,data in loop:
            inputs = data['utterances_tokenized'].to(self.device, dtype = torch.float32)
            triggers = data['triggers'].to(self.device, dtype = torch.float32)
            
            outputs = self.model(inputs, labels=triggers)
            print("###################################")
            print(outputs)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            self.optimizer.zero_grad()
            #loss = loss_fn(outputs, targets)
            loss = outputs.loss

            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            # big_idx = outputs.cpu().detach().numpy()>=0.5
            # n_correct += calcuate_accu(big_idx, targets)
            # nb_tr_examples += len(targets) * len(targets[0])
            nb_tr_steps += 1

            # accu_step = (n_correct*100)/nb_tr_examples
            average_loss = total_loss / nb_tr_steps

            loop.set_description(f'Epoch {epoch + 1}/{self.epochs}')
            #loop.set_postfix(loss_average = average_loss)
            #loop.set_postfix({'loss':loss.item(), 'loss_average':average_loss, 'accuracy':f'{accu_step:.2f}%'})

        return average_loss

    def validation():
        pass

    def train(self):
        self.model.train()

        train_losses = []
        val_losses = []

        # Early Stopping
        last_loss = 100
        patience = 2
        trigger_count = 0

        for epoch in range(self.EPOCHS):

            _train_loss = self.train_steps(epoch)
            train_losses.append(_train_loss)

            #@_val_loss = validation(model, validation_loader)
            #val_losses.append(_val_loss)

            # current_loss = _val_loss
            # if current_loss >= last_loss:
            #     trigger_count += 1

            # if trigger_count >= patience:
            #     print(f"Early Stopping has been triggred. Epoch: {epoch+1}")
            #     break
            # else:
            #     trigger_count = 0

            # last one or min ? difference is in the first if.
            #last_loss = current_loss

    def fineTuning():
        pass


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



    def train(self, model, training_loader, optimizer, epoch, epochs):

        total_loss = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        n_correct = 0

        loop = tqdm(enumerate(training_loader, 0), total=len(training_loader))
        for _,data in loop:

            targets = data['targets'].to(self.device, dtype = torch.float)
            outputs = model(data)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            big_idx = outputs.cpu().detach().numpy()>=0.5
            n_correct += calcuate_accu(big_idx, targets)
            nb_tr_examples += len(targets) * len(targets[0])
            nb_tr_steps += 1

            accu_step = (n_correct*100)/nb_tr_examples
            average_loss = total_loss / nb_tr_steps

            loop.set_description(f'Epoch {epoch + 1}/{epochs}')
            #loop.set_postfix(loss_average = average_loss)
            loop.set_postfix({'loss':loss.item(), 'loss_average':average_loss, 'accuracy':f'{accu_step:.2f}%'})

        return average_loss

    def validation():
        pass

    def trainer():
        pass

    def fineTuning():
        pass


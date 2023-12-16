# ==========================================
# Train and Validation Class
# ===========================================

from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
from utils.models import save_model, load_model

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

class EFRTraining():
    def __init__(self, training_loader, validation_loader, test_loader, device, seed=42, epochs=20, is_unfrozen=True):
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.seed = seed
        self.is_unforzen = is_unfrozen


    def train_steps(self, model, optimizer, epoch):
        model.train()

        total_loss = 0
        nb_tr_steps = 0
        all_preds = []
        all_labels = []

        loop = tqdm(enumerate(self.training_loader, 0), total=len(self.training_loader))
        for _,data in loop:
            utterances_input_ids = data['utterances_input_ids']
            targets = data['triggers']
            print(f"input shape: {utterances_input_ids.shape}")
            print(f"triggers shape: {targets.shape}")
            outputs = self.model(data)
            print("###################################")
            print(outputs)
            self.optimizer.zero_grad()
            #loss = loss_fn(outputs, targets)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            predictions = torch.argmax(outputs.logits,dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

            nb_tr_steps += 1
            avg_loss = total_loss / nb_tr_steps
            avg_accuracy = accuracy_score(all_labels, all_preds)

            loop.set_description(f'Epoch {epoch + 1}/{self.epochs}')
            #loop.set_postfix(loss_average = average_loss)
            loop.set_postfix({'loss':loss.item(), 'loss_average':avg_loss, 'accuracy':f'{avg_accuracy:.2f}%'})

        return avg_loss


    def validation(self, model):
        self.model.eval()
        nb_val_steps = 0
        total_loss = 0

        with torch.no_grad():
            loop = tqdm(enumerate(self.validation_loader, 0), total=len(self.validation_loader))

            for _, data in loop:
                outputs = model(data)

                loss = outputs.loss
                total_loss += loss.item()
                nb_val_steps += 1

                loop.set_description(f'validation: {nb_val_steps}/{len(self.validation_loader)}')

            average_loss = total_loss / nb_val_steps

        return average_loss


    def train(self, model, optimizer):
        
        
        train_losses = []
        val_losses = []

        # Early Stopping
        last_loss = 100
        patience = 2
        trigger_count = 0

        for epoch in range(self.epochs):

            _train_loss = self.train_steps(model, optimizer, epoch)
            train_losses.append(_train_loss)

            _val_loss = self.validation(model)
            val_losses.append(_val_loss)

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
        
        save_model(model, self.tokenizer,self.seed,"unforzen" if self.is_unforzen else "frozen")


    def test(self, model):
        model.eval()
        nb_test_steps = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            loop = tqdm(enumerate(self.validation_loader, 0), total=len(self.validation_loader))

            for _, data in loop:
                outputs = self.model(data)
                targets = data['triggers']

                predictions = torch.argmax(outputs.logits,dim=1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                nb_test_steps += 1

                avg_accuracy = accuracy_score(all_labels, all_preds)
                loop.set_description(f'Test: {nb_test_steps}/{len(self.validation_loader)}')
                loop.set_postfix({'accuracy':f'{avg_accuracy:.2f}%'})

        return all_preds, all_labels




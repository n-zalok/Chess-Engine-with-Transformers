import os 
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup
import mlflow
import mlflow.pytorch
# model's design
import architecture as arch


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# loading the dataset
ds = load_dataset("noor-zalouk/tournament-chess-games-modified")
print("Dataset loaded successfully.")

# aligning dataset with expected format
before = len(ds['train']) + len(ds['valid'])
def filter(example):
    if example['moves'][0:2] in arch.labels and example['moves'][2:4] in arch.labels:
        return True
    return False

ds = ds.filter(filter, num_proc=4)
after = len(ds['train']) + len(ds['valid'])
print(f"Removed {before-after} rows")

# data loader
class ChessDataset(Dataset):
    def __init__(self, ds, tokenizer, label_to_id, max_length):
        self.ds = ds
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        text = '[CLS]' + ' ' + self.ds[idx]['positions']
        input_ids = [self.tokenizer[word] for word in text.split()]
        input_ids = input_ids[:self.max_length]

        if self.ds[idx]['moves'] in ['e1g1', 'e8g8', 'e1c1', 'e8c8']:
            start_labels = self.label_to_id[self.ds['moves'][idx]]
            end_labels = start_labels
        else:
            start_labels = self.label_to_id[self.ds[idx]['moves'][0:2]]
            end_labels = self.label_to_id[self.ds[idx]['moves'][2:4]]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'start_labels': torch.tensor(start_labels, dtype=torch.long),
            'end_labels': torch.tensor(end_labels, dtype=torch.long)
        }
    
# initializing the model and the tokenizer
config = arch.Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = arch.tokenizer
label_to_id = arch.label_to_id
model = arch.ChessMoveClassifier(config, device)
model.to(device)

# training arguments
batch_size = 128
gradient_accumulation_steps = 1
epochs = 15
lr = 8e-4

# loss
criterion = nn.CrossEntropyLoss()

# data loaders
train_set = ChessDataset(ds['train'], tokenizer, label_to_id, config.max_position_embeddings)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
valid_set = ChessDataset(ds['valid'], tokenizer, label_to_id, config.max_position_embeddings)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0)

# define optimizer
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

# scheduler setup
num_update_steps_per_epoch = len(train_loader) // gradient_accumulation_steps
num_training_steps = num_update_steps_per_epoch * epochs
num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_training_steps=num_training_steps,
    num_warmup_steps=num_warmup_steps
)

# validation function
def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            start_labels = batch['start_labels'].to(device)
            end_labels = batch['end_labels'].to(device)

            start_logits, end_logits = model(input_ids=input_ids)
            loss_start = criterion(start_logits, start_labels)
            loss_end = criterion(end_logits, end_labels)
            loss = (loss_start + loss_end) / 2
            
            total_loss += loss.item()
            total_batches += 1

    avg_val_loss = total_loss / total_batches
    return avg_val_loss



# start an MLflow run
with mlflow.start_run():
    # log hyperparameters
    mlflow.log_params({
        "epochs": epochs,
        "batch_size": batch_size,
        "initial_lr": lr,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_warmup_steps": num_warmup_steps,
        "config": config.to_dict()
    })

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for step, batch in enumerate(loop):
            # get input_ids and labels then move them to device
            input_ids = batch['input_ids'].to(device)
            start_labels = batch['start_labels'].to(device)
            end_labels = batch['end_labels'].to(device)

            # get model's prediction
            start_logits, end_logits = model(input_ids=input_ids)

            # calculate the loss of start squares and end squares then take the average 
            loss_start = criterion(start_logits, start_labels)
            loss_end = criterion(end_logits, end_labels)
            loss = (loss_start + loss_end) / 2

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_batches += 1
            # show the current loss and learning rate
            loop.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

        train_loss = total_loss / total_batches
        # get validation loss
        valid_loss = evaluate(model, valid_loader, device)

        # log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": valid_loss,
            "lr": scheduler.get_last_lr()[0]
        }, step=epoch)

        # save model checkpoint with MLflow
        mlflow.pytorch.log_model(model, f"chess-epoch-{epoch}")
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")

print("Training complete. Model saved with MLflow.")

# saving the model
torch.save(model.state_dict(), "./chess_model.pth")
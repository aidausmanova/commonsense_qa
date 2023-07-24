import torch
from transformers import T5ForConditionalGeneration, AdamW, set_seed
from accelerate import Accelerator
from tqdm.notebook import tqdm
import datasets
import transformers
from torch.utils.data import DataLoader
from utils.services import read_data
from new_mlm import MaskedLMDataSet


hyperparameters = {
    "train_path":'/export/home/0usmanov/data/tsinghua_commonsense/train.txt',
    "val_path":'/export/home/0usmanov/data/tsinghua_commonsense/val.txt',
    "learning_rate": 0.0001,
    "num_epochs": 5, # set to very high number
    "train_batch_size": 16, 
    "eval_batch_size": 16,
    "seed": 42,
    "patience": 3, # early stopping
    "output_dir": "/export/home/0usmanov/project/output/code_encoder/",
}


def create_dataloaders(train_batch_size=8, eval_batch_size=32):
    train_dataloader = DataLoader(encoded_train_ds, shuffle=True, batch_size=train_batch_size)
    val_dataloader = DataLoader(encoded_val_ds, shuffle=False, batch_size=eval_batch_size)
    
    return train_dataloader, val_dataloader


def training_function():
    set_seed(hyperparameters["seed"])

    # Instantiate the model, let Accelerate handle the device placement.
    tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Instantiate optimizer
    optimizer = AdamW(model.parameters(), lr=hyperparameters["learning_rate"])

    train_lines = read_data(hyperparameters["train_path"])
    val_lines = read_data(hyperparameters["val_path"])
    train_dataset = MaskedLMDataSet(train_lines, 0.15, tokenizer, 256)
    val_dataset = MaskedLMDataSet(val_lines, 0.15, tokenizer, 256)

    # Prepare everything
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=hyperparameters["train_batch_size"]) 
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=hyperparameters["train_batch_size"])

    # model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, 
    #                                                                          train_dataloader, val_dataloader)
    

    epochs_no_improve = 0
    min_val_loss = 1000000
    for epoch in range(hyperparameters["num_epochs"]):
        # We only enable the progress bar on the main process to avoid having 8 progress bars.
        progress_bar = tqdm(range(len(train_dataloader)))
        progress_bar.set_description(f"Epoch: {epoch}")
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({'loss': loss.item()})
            progress_bar.update(1)

        # Evaluate at the end of the epoch (distributed evaluation as we have 8 TPU cores)
        model.eval()
        validation_losses = []
        for batch in val_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss

            # We gather the loss from the 8 TPU cores to have them all.
            validation_losses.append(loss[None])

        # Compute average validation loss
        val_loss = torch.stack(validation_losses).sum().item() / len(validation_losses)
        # Use accelerator.print to print only on the main process.
       print(f"epoch {epoch}: validation loss:", val_loss)
        if val_loss < min_val_loss:
          epochs_no_improve = 0
          min_val_loss = val_loss
          continue
        else:
          epochs_no_improve += 1
          # Check early stopping condition
          if epochs_no_improve == hyperparameters["patience"]:
            print("Early stopping!")
            break

    model_dir = '/export/home/0usmanov/project/output/code_encoder/checkpoints/'
    torch.save(model.state_dict(), model_dir+f"t5s_adamw_pretrain_{local_start_time_str}.ckpt")
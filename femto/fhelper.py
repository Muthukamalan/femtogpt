import torch
from typing import Optional
import glob
import re
import logging
from datasets import IterableDatasetDict,IterableDataset

def summarize_text(text: str, n: int = 10) -> str:
    words = text.split()
    if len(words) <= 2 * n:return text  # no need to shorten
    start = words[:n]
    end = words[-n:]
    return " ".join(start) + " .... " + " ".join(end)

def display_model_summary(model:torch.nn.Module):
    """Display model parameters summary including total, trainable params and model size"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info("Model Parameters Summary:")
    logging.info(f"{'='*50}")
    logging.info(f"Total Parameters:     {total_params / 2**20:.2f} Million(s)")
    logging.info(f"Trainable Parameters: {trainable_params:,}")
    logging.info(f"Model Size:           {total_params * 4 / (1024**2):.2f} MB")
    logging.info(f"{'='*50}")


def save_checkpoint(model:torch.nn.Module, optimizer:torch.optim.Optimizer, scheduler:torch.optim.lr_scheduler.StepLR, step:int,  path:str):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, path)



def load_checkpoint(path:str, model:torch.nn.Module, optimizer=None, scheduler=None):
    print(f"Loading checkpoint from {path}, Please wait...")
    checkpoint = torch.load(path, weights_only=False)  # Explicitly set weights_only
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['step'], checkpoint['loss']



def get_latest_checkpoint(dir) -> tuple[Optional[str], Optional[int]]:
    """Returns tuple of (checkpoint_path, step_number) or (None, None) if no checkpoints exist

    `note`: searched in patter `model_step_(??)_loss_(??).pt`
    """
    checkpoints = glob.glob(f"{dir}/*.pt")
    if not checkpoints:
        return None, None
    
    # Print all checkpoints for debugging
    print(f"\nTotal checkpoints found: {len(checkpoints)}")
    print("Extracting steps from checkpoints...")
    
    steps = []
    for ckpt in checkpoints:
        # Extract step number from filename
        match = re.search(r'model_step_(\d+)_loss_', ckpt)
        if match:
            step_num = int(match.group(1))
            steps.append((step_num, ckpt))
            print(f"+ Extracted step {step_num} from {ckpt}")
    
    if steps:
        latest_step, latest_ckpt = max(steps, key=lambda x: x[0])
        print("\nSelected checkpoint:")
        print(f"  Path: {latest_ckpt}")
        print(f"  Step: {latest_step}")
        return latest_ckpt, latest_step
    return None, None


def num_parameters(module: torch.nn.Module, requires_grad: Optional[bool] = None) -> int:
    return sum(
        p.numel()
        for p in module.parameters()
        if requires_grad is None or p.requires_grad == requires_grad
    )





def split_streaming_dataset(
    full_streaming_dataset,
    validation_percentage: int = 5,
) -> IterableDatasetDict:
    """
    Splits a streaming dataset into
    training and validation IterableDatasets, and supports methods like .map(), .filter(),
    .take() and properties like .features on the resulting streams.

    Args:
        full_streaming_dataset (Dataset): The name of the dataset to load (e.g., "HuggingFaceFW/fineweb").
        validation_percentage (int): The proportion of the dataset to be used for validation split.

    Returns:
        IterableDatasetDict: An IterableDatasetDict containing two IterableDataset objects: (train_stream, validation_stream).
    """
    if not (0 < validation_percentage < 100): raise ValueError(f"validation_percentage must be between 0 and 100 (exclusive). Passed: {validation_percentage}"        )

    def split_generator(is_train: bool):
        for i, example in enumerate(full_streaming_dataset):
            if is_train:
                if i % 100 > validation_percentage: yield example
            else:
                if i % 100 < validation_percentage: yield example

    features = full_streaming_dataset.features
    train_stream = IterableDataset.from_generator(split_generator, gen_kwargs={"is_train": True}, features=features)
    validation_stream = IterableDataset.from_generator(split_generator, gen_kwargs={"is_train": False}, features=features )
    return IterableDatasetDict({"train": train_stream, "validation": validation_stream})


# streamin_ds = eval_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", streaming=True)
# raw_dataset = split_streaming_dataset(streamin_ds)



def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("pre-train.log")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
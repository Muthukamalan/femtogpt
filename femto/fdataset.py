import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, interleave_datasets

# inspire from [infinite-ds](https://github.com/huggingface/transformers/blob/resnet_with_variants/examples/research_projects/codeparrot/scripts/codeparrot_training.py)
class StreamingDataset(IterableDataset):
    def __init__(self, tokenizer, block_size=512,mode:str='train'):
        self.block_size = block_size
        self.tokenizer = tokenizer

        # Add high-quality datasets
        datasets = [
            load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True,split=mode),
            load_dataset("databricks/databricks-dolly-15k", streaming=True,split=mode),
            load_dataset("Abirate/english_quotes", streaming=True,split=mode),
            load_dataset("b-mc2/sql-create-context", streaming=True,split=mode),
            load_dataset("squad", streaming=True,split=mode),
            load_dataset("tatsu-lab/alpaca", streaming=True,split=mode)  # Instruction dataset
        ]

        # Adjust weights for better balance
        self.dataset = interleave_datasets(
            datasets,
            probabilities=[
                0.35,
                0.2,
                0.1,
                0.1,
                0.1,
                0.15,
            ],  # Adjusted weights for 6 datasets
            stopping_strategy="first_exhausted",
        )

        # More aggressive quality filtering
        def quality_filter(example):
            text = (
                example.get("text")
                or example.get("content")
                or example.get("instruction")
                or example.get("question")
                or example.get("context")
                or ""
            )
            # Stricter quality checks
            return (
                len(text) > 200  # Longer texts
                and text.count(".") >= 2  # Multiple sentences
                and len(set(text.split())) > 50  # More vocabulary diversity
            )

        self.dataset = self.dataset.filter(quality_filter)

        # Enable shuffling with buffer
        self.dataset = self.dataset.shuffle(seed=46, buffer_size=10000)

    def __iter__(self):
        iterator = iter(self.dataset)
        buffer = []

        while True:
            try:
                # Get next example
                example = next(iterator)

                # Handle different text field names
                text = (
                    example.get("text")
                    or example.get("content")
                    or example.get("instruction")
                    or example.get("code")
                    or ""
                )


                # Train Iterable Dataset Over Trainer API
                yield self.tokenizer(text.strip(),truncation=True,max_length=self.block_size,padding=False)

                # # Tokenize text
                # tokens = self.tokenizer.encode(text)

                # # Add to buffer
                # buffer.extend(tokens)

                # # Process buffer into chunks
                # while len(buffer) >= self.block_size:
                #     chunk = buffer[: self.block_size]
                #     buffer = buffer[self.block_size :]

                #     # Create input and target sequences
                #     x = torch.tensor(chunk[:-1], dtype=torch.long)
                #     y = torch.tensor(chunk[1:], dtype=torch.long)

                #     yield x, y

            except StopIteration:
                # Reset iterator when exhausted
                iterator = iter(self.dataset)
                buffer = []

def collate_batch(batch):
    """Custom collate function to pad sequences in a batch.
    batch:: 
    [(x1,y1),(x2,y2),......(x_bs,y_bs)]
    input_seq,target_seq = map(list, zip(*batch))

    returns:
    inputs:
    [x1,x2,..,x32]
    target:
    [y1,y2,...,y32]
    """
    # Batch from IterableDataset
    input_seq,target_seq = map(list, zip(*batch))
    # Find max length in the batch
    max_len = max(len(x) for x in input_seq)
    # Initialize tensors for inputs and targets
    batch_size = len(batch)
    inputs = torch.full((batch_size, max_len), fill_value=0, dtype=torch.long)  # 0 is assumed to be pad_token_id
    targets = torch.full((batch_size, max_len), fill_value=-100, dtype=torch.long)  # -100 is ignored by CrossEntropyLoss
    # Fill in the tensors with actual values
    for i, (input_seq, target_seq) in enumerate(zip(input_seq,target_seq)):
        seq_len = input_seq.size(0)
        inputs[i, :seq_len] = input_seq
        targets[i, :seq_len] = target_seq
    return inputs, targets
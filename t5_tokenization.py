from datasets import load_from_disk
from transformers import AutoTokenizer

# load the cleaned saved HuggingFace Dataset Dictionary
cleaned_dataset = load_from_disk("bhc_cleaned_dataset")

# set the T5 model and appropriate AutoTokenizer
model_name = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# set the max input and target lengths
max_input_length = 1024
max_target_length = 256

# tokenize the bodies and summaries from the notes for the T5 model
def preprocess_t5(notes):
    # tokenize the inputs
    model_inputs = tokenizer(
        notes["body"],
        max_length=max_input_length,
        truncation=True, # truncate if longer than max length
        padding="max_length" # pad to max length for consistency
    )

    # tokenize the targets
    labels = tokenizer(
        notes["summary"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )

    # set the labels for the inputs as the ids for the summaries
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# map the tokenization to all the data splits, drop original column names
tokenized_dataset = cleaned_dataset.map(
    preprocess_t5,
    batched=True, # speed up tokenization by feeding batches of notes
    remove_columns=["body", "summary"]
)

# save tokenized dataset to disk
tokenized_dataset.save_to_disk("t5_small_tokenized")
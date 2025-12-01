from transformers import T5ForConditionalGeneration, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import load_from_disk

# set the model and tokenizer
model_name = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# get the saved tokenized dataset
tokenized_path = "./t5_tokenized"
dataset = load_from_disk(tokenized_path)

train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

# set up the data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model, 
    padding="longest",
    label_pad_token_id=tokenizer.pad_token_id
)

# set up training arguments
training_args = TrainingArguments(
    output_dir="./t5_checkpoints",
    per_device_train_batch_size=8, # small batch sizes due to long inputs
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    num_train_epochs=3,
    logging_steps=50,
    eval_strategy="epoch", # eval after every epoch
    save_strategy="epoch",
    fp16=False,
    push_to_hub=False
)

# set up the trainer with the model, datasets, arguments, data collator, and tokenizer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# train the model
trainer.train()

trainer.save_model("./t5_model")
tokenizer.save_pretrained("./t5_model")

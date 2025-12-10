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

# set up the data collator that will prepare the batches for the T5 model
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model, 
    padding=False,
    label_pad_token_id=-100
)

# set up training arguments
training_args = TrainingArguments(
    output_dir="./t5_checkpoints",
    per_device_train_batch_size=8, # smaller batch sizes due to long inputs
    per_device_eval_batch_size=8, 
    learning_rate=5e-5, # slightly faster learning rate for T5 small for large data size
    num_train_epochs=3, # run all batches 3 times
    logging_steps=50, # log metrics after 50 steps (batches)
    eval_strategy="epoch", # evaluation after every epoch
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

# save the trained model and tokenizer
trainer.save_model("./t5_model")
tokenizer.save_pretrained("./t5_model")

from transformers import T5ForConditionalGeneration, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Trainer
from datasets import load_from_disk
import evaluate
import torch
from tqdm import tqdm 

# load the trained model and tokenizer
model_path = "./t5_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# load the tokenized dataset and rouge evaluation metric
dataset = load_from_disk("./t5_tokenized")
test_data = dataset["test"]
rouge = evaluate.load("rouge")

# set up the decoding pipeline
def postprocess_text(preds, labels):
    # decode the token ids to strings for the generated predictions and labels
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # strip whitespace
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    return preds, labels

# set the model in evaluation mode and move model to GPU/CPU for input tensors
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

all_preds = []
all_labels = []

batch_size = 8  
# evaluate the test set
for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
    batch = test_data[i : i + batch_size]
    
    # convert batch to tensors on the selected device
    input_ids = torch.tensor(batch["input_ids"]).to(device)
    attention_mask = torch.tensor(batch["attention_mask"]).to(device)
    labels = torch.tensor(batch["labels"]).to(device)
    
    # generate predictions, disable gradient computation
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,  # same as target length
            num_beams=2, # explore different options 
            length_penalty=1.2, # slightly favor longer sequences
            no_repeat_ngram_size=3, # reduce repetitive text  
            early_stopping=True
        )
    
    # store predictions and labels in a list
    all_preds.extend(outputs)
    all_labels.extend(labels)

# decode predictions and references
decoded_preds, decoded_labels = postprocess_text(all_preds, all_labels)

# print samples of generated examples and their labels
print("\n--- Sample generated examples ---")
for i in range(5):
    print(f"\nExample {i+1}:")
    print(f"Reference : {decoded_labels[i]}")
    print(f"Prediction: {decoded_preds[i]}")

# compute ROUGE scores as percentages
result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
result = {key: value * 100 for key, value in result.items()}

# print ROUGE scores
print("ROUGE scores on the test set:")
for k, v in result.items():
    print(f"{k}: {v:.2f}")

import os
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import RobertaTokenizer,RobertaForMaskedLM,RobertaConfig

from DataCollatorForLanguageModeling import DataCollatorForLanguageModeling
from LineByLineTextDataset import LineByLineTextDataset
from transformers import Trainer, TrainingArguments

# Building a Tokenizer
# Then we have two files that define our new EsperBERTo tokenizer:
# merges.txt — performs the initial mapping of text to tokens
# vocab.json — maps the tokens to token IDs


paths = ['./oscar-small.eo.txt']
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
	"<s>",
	"<pad>",
	"</s>",
	"<unk>",
	"<mask>",
])
if not os.path.exists('./EsperBERToX'):
	os.mkdir('./EsperBERToX')
tokenizer.save_model("./EsperBERToX")

# load tokenizer
tokenizer = RobertaTokenizer.from_pretrained('./EsperBERToX')


config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)


# block_size: max length of batch
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./oscar-small.eo.txt",
    block_size=200,
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

model = RobertaForMaskedLM(config=config)


training_args = TrainingArguments(
    output_dir="./EsperBERTo",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_gpu_train_batch_size=12,
    save_steps=2000,
    save_total_limit=2,
    learning_rate=1e-4,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    # prediction_loss_only=True,
)

trainer.train()
trainer.save_model("./EsperBERTo")

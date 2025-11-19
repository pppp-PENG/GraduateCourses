#!/usr/bin/env python
# coding: utf-8

# # Fine-tune QWEN in Google Colab

# This notebook guide provides a comprehensive overview of using the `transformers` Python package to efficiently train a custom model. It covers the following techniques:
#
# 1. Utilizing model, tokenizer, and dataset loading functionalities from Hugging Face.
# 2. Performing basic data cleaning.
# 3. Training the model with basic modeling techniques, including quantization, such as qlora in this instance.
# 4. Evaluating the model's performance on test set.
# 5. Saving your custom model and preparing it for deployment.

# ## Preliminary Preparation
#
# Before proceeding with model training, ensure your environment is properly configured by following these steps:
#
# 1. Install the necessary Python packages.
# 2. Import the required libraries.

# In[1]:


# !pip install -q h5py typing-extensions wheel fschat
# # !pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 fschat
# !pip install -q -U bitsandbytes
# !pip install -q -U git+https://github.com/huggingface/transformers.git
# !pip install -q -U git+https://github.com/huggingface/peft.git
# !pip install -q -U git+https://github.com/huggingface/accelerate.git
# !pip install -q datasets


# In[2]:


get_ipython().system("nvidia-smi")


# ## Load Pre-trained model and tokenizer

# First let's load the model we are going to use - phoenix-inst-chat-7b! Note that the model itself is around 7B in full precision

# In[3]:


import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/model/"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

DataAugmentation = True
train_dataset_size, val_dataset_size = 400, 100


# Quantization type (fp4 or nf4), According to QLoRA paper, for training 4-bit base models (e.g. using LoRA adapters) one should use
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = True

model_id = "Qwen/Qwen3-14B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=use_nested_quant,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Show special tokens
print(f"Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
print(f"BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
print(f"UNK token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map={"": 0}
)


# Then we have to apply some preprocessing to the model to prepare it for training. For that use the `prepare_model_for_kbit_training` method from PEFT.

# In[4]:


from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# Lora Configuration
config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


# In[5]:


if myDebug:
    # from fastchat.conversation import get_conv_template
    TRANSFORMERS_VERBOSITY = "info"
    device = "cuda"
    model.eval()

    @torch.no_grad()
    def generate(prompt):
        input_ids = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        outputs = model.generate(input_ids, do_sample=False, max_new_tokens=1024)
        return tokenizer.decode(*outputs, skip_special_tokens=True)

    prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
        add_generation_prompt=True,
    )
    response = generate(text)
    print("-" * 80)
    print(response)


# ## Data Preparation

# Let's load a common dataset, english quotes, to fine tune our model on famous quotes.

# In[6]:


from datasets import load_dataset

# data = load_dataset("Abirate/english_quotes")
dataset = load_dataset("FreedomIntelligence/Huatuo26M-Lite")
dataset = dataset["train"].map(
    lambda sample: {
        "conversations": [
            {"role": "human", "value": sample["question"]},
            {"role": "gpt", "value": sample["answer"]},
        ]
    },
    batched=False,
)


# In[7]:


from torch.utils.data import random_split


# In[8]:


train_dataset, val_dataset, _ = random_split(
    dataset,
    [
        train_dataset_size,
        val_dataset_size,
        len(dataset) - train_dataset_size - val_dataset_size,
    ],
)


# ## Data Augmentation

# In[9]:


if DataAugmentation:
    from openai import OpenAI
    import time
    import random
    from torch.utils.data import Dataset, ConcatDataset

    # init client
    client = OpenAI(
        api_key="sk-5f2d8dd984544662bd4177b4f80b41b3",
        base_url="https://api.deepseek.com/v1",
    )

    def augment_with_deepseek(text, max_retries=3):
        prompt = f"""
        请对以下文本进行同义词替换，保持原意不变，只返回替换后的文本：

        原文：{text}

        增强后的文本：
        """

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的文本增强助手，擅长进行同义词替换而不改变原文意思。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=len(text) + 100,
                    temperature=0.7,
                    timeout=30,
                )

                augmented_text = response.choices[0].message.content.strip()

                # clear the prefix
                if "增强后的文本：" in augmented_text:
                    augmented_text = augmented_text.split("增强后的文本：")[-1].strip()

                # ensure that the augmented text is different from the original
                if augmented_text and augmented_text != text:
                    return augmented_text
                else:
                    # if augmentation fails, return the original text with slight modification
                    words = text.split()
                    if len(words) > 3:
                        # randomly replace one word with a placeholder
                        replace_idx = random.randint(0, len(words) - 1)
                        words[replace_idx] = f"[同义词替换]{words[replace_idx]}"
                        return " ".join(words)
                    return text

            except Exception as e:
                print(f"Using API false (retry {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"All retries failed, returning original text: {text}")
                    return text

    def augment_dataset(dataset, augmentation_ratio=0.3):
        """
        augmentation_ratio: proportion of data to be augmented
        """
        augmented_data = []

        # randomly select samples to augment
        total_samples = len(dataset)
        num_augment = int(total_samples * augmentation_ratio)
        indices_to_augment = random.sample(range(total_samples), num_augment)

        print(f"start processing {num_augment} samples...")

        for i, idx in enumerate(indices_to_augment):
            if i % 10 == 0:
                print(f"progress: {i}/{num_augment}")
            sample = dataset[idx]
            original_question = sample["conversations"][0]["value"]

            # augment question
            augmented_question = augment_with_deepseek(original_question)

            # create augmented sample
            augmented_sample = {
                "conversations": [
                    {"role": "human", "value": augmented_question},
                    {"role": "gpt", "value": sample["conversations"][1]["value"]},
                ]
            }

            augmented_data.append(augmented_sample)

            # # Add delay to avoid API rate limits
            # time.sleep(1)

        return augmented_data

    # Execute data augmentation
    print("Start data augmentation...")
    augmented_samples = augment_dataset(train_dataset, augmentation_ratio=0.3)

    # Convert augmented data to dataset format
    class AugmentedDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    augmented_dataset = AugmentedDataset(augmented_samples)

    # Merge original training set and augmented dataset
    train_dataset = ConcatDataset([train_dataset, augmented_dataset])
    print(f"Training set size after augmentation: {len(train_dataset)}")


# ### Customized Dataset
# Create a specialized dataset class named "InstructionDataset" designed to handle our custom dataset.

# In[10]:


import json, copy
import transformers
from typing import Dict, Sequence, List
from dataclasses import dataclass
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class InstructDataset(Dataset):
    def __init__(
        self, data: Sequence, tokenizer: transformers.PreTrainedTokenizer
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        sources = self.data[index]
        if isinstance(index, int):
            sources = [sources]
        data_dict = preprocess([e["conversations"] for e in sources], self.tokenizer)
        if isinstance(index, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )
        return data_dict


def preprocess(
    sources: Sequence[Sequence[Dict]],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length=1024,
) -> Dict:
    input_ids_list = []
    labels_list = []

    for source in sources:
        messages = []
        for msg in source:
            if msg["role"].lower() == "human":
                role = "user"
            elif msg["role"].lower() == "gpt":
                role = "assistant"
            else:
                role = "user"
            messages.append({"role": role, "content": msg["value"]})

        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False, add_generation_prompt=False
        )

        # Tokenize
        input_ids = tokenizer.encode(formatted, add_special_tokens=False)

        labels = [IGNORE_INDEX] * len(input_ids)

        text_parts = formatted.split("<|im_start|>")
        assistant_start_pos = 0

        for i, part in enumerate(text_parts):
            if part.startswith("assistant"):
                prev_parts = "<|im_start|>".join(text_parts[:i])
                if prev_parts:
                    prev_parts += "<|im_start|>"
                assistant_start_pos = len(prev_parts) + len("assistant\n")
                break

        if assistant_start_pos > 0:
            prefix_text = formatted[:assistant_start_pos]
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

            if len(prefix_ids) < len(input_ids):
                for i in range(len(prefix_ids), len(input_ids)):
                    labels[i] = input_ids[i]

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        input_ids_list.append(torch.tensor(input_ids))
        labels_list.append(torch.tensor(labels))

    return dict(input_ids=input_ids_list, labels=labels_list)


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# In[11]:


train_dataset = InstructDataset(train_dataset, tokenizer)
val_dataset = InstructDataset(val_dataset, tokenizer)
data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

# ## Training

# ### General Training Hyperparameters

# In[14]:


# Set training parameters
training_arguments = transformers.TrainingArguments(
    output_dir="./checkpoint",
    num_train_epochs=1,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=1,
    learning_rate=1e-4,
    weight_decay=0.001,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    gradient_checkpointing=False,
    report_to="none",
    eval_strategy="steps",
    eval_steps=1,
    # load_best_model_at_end=True,
    # metric_for_best_model="eval_loss",
)

from transformers import TrainerCallback
import json


class LossCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.output_file = "./training_losses.json"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_losses.append(
                {"step": state.global_step, "loss": logs["loss"], "epoch": state.epoch}
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            self.eval_losses.append(
                {
                    "step": state.global_step,
                    "eval_loss": metrics["eval_loss"],
                    "epoch": state.epoch,
                }
            )

    def on_train_end(self, args, state, control, **kwargs):
        loss_data = {"train_losses": self.train_losses, "eval_losses": self.eval_losses}
        with open(self.output_file, "w") as f:
            json.dump(loss_data, f, indent=2)
        print(f"Training loss has saved to: {self.output_file}")


loss_callback = LossCallback()


# In[15]:


model.train()
trainer = transformers.Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[loss_callback],
)

model.print_trainable_parameters()


# In[16]:


trainer.train()


# Once the training is completed, we can evaluate our model and get its perplexity on the validation set like this:

# In[17]:


import math

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


# ## Save Trained LoRA

# In[18]:


output_path = "trained_model"
trainer.save_model(output_path)

# In[20]:


# Empty VRAM
del model
del trainer
import gc

gc.collect()
gc.collect()


# ## Load the trained model back and integrate the trained LoRA within.

# In[21]:


from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Re-define quantization parameters if needed, or use the ones defined earlier
bnb_4bit_quant_type = "nf4"  # or "fp4"
use_nested_quant = True  # or False

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=use_nested_quant,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map={"": 0}
)
model = PeftModel.from_pretrained(model, "./trained_model")
model = model.merge_and_unload()
model.config.max_length = 512
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
# tokenizer.pad_token = tokenizer.unk_token # This line is not needed and can be removed


# ## Answer generation

# In[22]:


from tqdm import tqdm


def generate(query_list, return_answer: bool = False):
    messages_list = []
    for query in query_list:
        messages = [{"role": "user", "content": query}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False, add_generation_prompt=True
        )
        messages_list.append(formatted)

    # Tokenize
    inputs = tokenizer(
        messages_list,
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = inputs.input_ids.to("cuda")

    n_input, n_seq = input_ids.shape[0], input_ids.shape[-1]
    output_ids = []
    step = 1

    for index in tqdm(range(0, n_input, step)):
        outputs = model.generate(
            input_ids=input_ids[index : min(n_input, index + step)],
            do_sample=True,
            max_new_tokens=800,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            early_stopping=True,
        )
        output_ids += outputs

    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    if return_answer:
        answers = []
        for prompt, response in zip(messages_list, responses):
            if response.startswith(prompt):
                raw_answer = response[len(prompt) :].strip()

                if raw_answer.startswith("assistant") or raw_answer.startswith(
                    "assistant\n"
                ):
                    lines = raw_answer.split("\n", 1)
                    if len(lines) > 1:
                        answer = lines[1].strip()
                    else:
                        answer = raw_answer.replace("assistant", "").strip()
                else:
                    answer = raw_answer
            else:
                assistant_markers = ["assistant\n", "assistant:"]
                for marker in assistant_markers:
                    if marker in response:
                        parts = response.split(marker, 1)
                        if len(parts) > 1:
                            answer = parts[1].strip()
                            if "user\n" in answer:
                                answer = answer.split("user\n")[0].strip()
                            break
                else:
                    answer = response

            answer = answer.lstrip()
            answers.append(answer)

        return answers

    return responses


# Test generation
print(
    "\n".join(
        generate(["What's the weather like today?", "Who are you?"], return_answer=True)
    )
)


# ## Evaluate a trained model on a given test dataset

# In[23]:


import os

# TODO: correctly put test data files into an accessible path
test_file = "zh_med.json"
assert os.path.exists(test_file), "Invalid test_file path"

with open(test_file, "r", encoding="utf-8") as reader:
    test_data = json.load(reader)
# print(test_data[0])


# In[24]:


model_answers = generate([data[0] for data in test_data], return_answer=True)


# In[25]:


for data, answer in zip(test_data, model_answers):
    data.append(answer)


# In[26]:


with open("saved_data.json", "w", encoding="utf-8") as writer:
    json.dump(test_data, writer, indent=4, ensure_ascii=False)

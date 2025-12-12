# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs inference with a MetricX model."""

import dataclasses
import json
import os

import datasets
from metricx24 import models
import torch
import transformers
from transformers import DataCollatorWithPadding


@dataclasses.dataclass
class Arguments:
  """Prediction command-line arguments."""

  tokenizer: str = dataclasses.field(
      metadata={"help": "The name of the tokenizer"},
  )

  model_name_or_path: str = dataclasses.field(
      metadata={
          "help": (
              "Path to pretrained model or model identifier from"
              " huggingface.co/models"
          )
      },
  )

  max_input_length: int = dataclasses.field(
      metadata={"help": "The maximum allowable input sequence length."},
  )

  batch_size: int = dataclasses.field(
      metadata={"help": "The global prediction batch size."},
  )

  input_file: str = dataclasses.field(metadata={"help": "The input file."})

  output_file: str = dataclasses.field(
      metadata={"help": "The output file with predictions."},
  )

  qe: bool = dataclasses.field(
      metadata={"help": "Indicates the metric is a QE metric."},
      default=False,
  )


def get_dataset(
    input_file: str, tokenizer, max_input_length: int, device, is_qe: bool
):
    """Gets the test dataset for prediction."""

    def _make_input(example):
        if is_qe:
            example["input"] = (
                "source: "
                + example["source"]
                + " candidate: "
                + example["hypothesis"]
            )
        else:
            example["input"] = (
                "source: "
                + example["source"]
                + " candidate: "
                + example["hypothesis"]
                + " reference: "
                + example["reference"]
            )
        return example

    def _tokenize(example):
        return tokenizer(
            example["input"],
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )

    def _remove_eos(example):
        example["input_ids"] = example["input_ids"][:-1]
        example["attention_mask"] = example["attention_mask"][:-1]
        return example

    ds = datasets.load_dataset("json", data_files={"test": input_file})

    # 1. Track original index immediately
    ds["test"] = ds["test"].map(
        lambda x, idx: {"original_index": idx},
        with_indices=True
    )

    # 2. Process text (Tokenize)
    ds = ds.map(_make_input)
    ds = ds.map(_tokenize)
    ds = ds.map(_remove_eos)

    # 3. Calculate length AFTER tokenization
    ds["test"] = ds["test"].map(lambda x: {"length": len(x["input_ids"])})

    # 4. Sort by length to boost speed (and flatten to prevent crashes)
    ds["test"] = ds["test"].sort("length", reverse=True).flatten_indices()

    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "original_index"],
        device=device,
        output_all_columns=True,
    )

    # Add data collator for batching mode
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    return ds, data_collator


def main() -> None:
    parser = transformers.HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        per_device_batch_size = args.batch_size // torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        per_device_batch_size = args.batch_size

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

    model = models.MT5ForRegression.from_pretrained(
        args.model_name_or_path, torch_dtype="auto"
    )

    model.to(device)
    model.eval()

    ds, datacollator = get_dataset(
        args.input_file,
        tokenizer,
        args.max_input_length,
        device,
        args.qe,
    )

    training_args = transformers.TrainingArguments(
        output_dir=os.path.dirname(args.output_file),
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_pin_memory=False,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=datacollator
    )
    
    # Predict on the SORTED dataset
    predictions, _, _ = trainer.predict(test_dataset=ds["test"])

    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # 5. Buffer results to memory to re-sort them
    results = []
    for pred, example in zip(predictions, ds["test"]):
        example["prediction"] = float(pred)
        
        # Clean up internal columns
        del example["input"]
        del example["input_ids"]
        del example["attention_mask"]
        del example["length"]
        # Note: We keep "original_index" temporarily
        
        results.append(example)

    # 6. RESTORE original order using the index we tracked
    results.sort(key=lambda x: x["original_index"])

    with open(args.output_file, "w") as out:
        for example in results:
            # Remove the tracking index before saving
            del example["original_index"]
            out.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    main()

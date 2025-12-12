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
  """Gets the test dataset for prediction.

  If `is_qe` is true, the input data must have "hypothesis" and "source" fields.
  If it is false, there must be "hypothesis" and "reference" fields.

  Args:
    input_file: The path to the jsonl input file.
    tokenizer: The tokenizer to use.
    max_input_length: The maximum input sequence length.
    device: The ID of the device to put the PyTorch tensors on.
    is_qe: Indicates whether the metric is a QE metric or not.

  Returns:
    The dataset.
  """

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
  ds = ds.map(_make_input)
  ds = ds.map(_tokenize)
  ds = ds.map(_remove_eos)
  ds.set_format(
      type="torch",
      columns=["input_ids", "attention_mask"],
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

    # Optimization 1: Compile the model (if PyTorch 2.0+)
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    ds, datacollator = get_dataset(
        args.input_file,
        tokenizer,
        args.max_input_length,
        device,
        args.qe,
    )

    # --- NEW LOGIC: Sort for Speed, Track Index for Order ---
    
    # 1. Add original index and length columns
    ds["test"] = ds["test"].map(
        lambda x, idx: {"original_index": idx, "length": len(x["input_ids"])},
        with_indices=True
    )

    # 2. Sort by length descending (minimizes padding overhead)
    sorted_test_ds = ds["test"].sort("length", reverse=True).flatten_indices()

    training_args = transformers.TrainingArguments(
        output_dir=os.path.dirname(args.output_file),
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_pin_memory=True,
        # Optimization 2: Use FP16 if on GPU
        fp16=torch.cuda.is_available(), 
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=datacollator
    )

    # 3. Run prediction on the SORTED dataset
    predictions, _, _ = trainer.predict(test_dataset=sorted_test_ds)

    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # 4. Collect results into a list
    results_buffer = []
    for pred, example in zip(predictions, sorted_test_ds):
        example["prediction"] = float(pred)
        
        # Cleanup internal columns before saving
        del example["input"]
        del example["input_ids"]
        del example["attention_mask"]
        del example["length"] 
        # Note: Keep 'original_index' for a moment to sort back
        
        results_buffer.append(example)

    # 5. RE-ORDER: Sort the results back to their original input order
    results_buffer.sort(key=lambda x: x["original_index"])

    # 6. Write to file
    with open(args.output_file, "w") as out:
        for example in results_buffer:
            # Remove the index now that we are done sorting
            del example["original_index"]
            out.write(json.dumps(example) + "\n")


if __name__ == "__main__":
  main()

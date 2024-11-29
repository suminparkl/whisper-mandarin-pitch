import argparse
import yaml
import torch
import numpy as np
import gc
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from transformers import (
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
)
from peft import PeftModel
from datasets import load_from_disk
import evaluate

from data_utils import (
    add_f0_as_additional_channel,
    prepare_dataset,
    DataCollatorSpeechSeq2SeqWithPadding,
)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate the Whisper model with pitch features")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the configuration file"
    )
    args = parser.parse_args()

    # Load configurations from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set evaluation metric
    metric = evaluate.load("wer")

    # Define processor
    model_name_or_path = config["model_name_or_path"]
    language = config["language"]
    task = config["task"]
    processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

    # Load dataset
    dataset = load_from_disk(config["eval_dataset_path"])
    test_data = dataset["test"]

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)

    # Load base model
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float32,
    )

    # Modify the first Conv1D layer to accept additional pitch channel
    from model_utils import modify_first_conv_layer
    model = modify_first_conv_layer(model, dtype=torch.float32)

    # Load PEFT model
    peft_model_id = config["peft_model_id"]
    peft_model = PeftModel.from_pretrained(model, peft_model_id)
    peft_model.config.use_cache = True
    peft_model.to("cuda")

    # Prepare dataset
    test_data = test_data.map(
        lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
        num_proc=config.get("num_proc", 4),
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    eval_dataloader = DataLoader(
        test_data,
        batch_size=config.get("eval_batch_size", 8),
        collate_fn=data_collator,
        num_workers=config.get("num_workers", 4),
        prefetch_factor=config.get("prefetch_factor", 2),
    )

    print(f"Language: {language}")
    print(f"Task: {task}")

    # Get decoder prompt ids
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)

    # Dictionary to track adapter layer usage
    module_usage = defaultdict(int)

    # Define forward hook function
    def create_forward_hook(name):
        def forward_hook(module, input, output):
            module_usage[name] += 1

        return forward_hook

    # Register hooks on adapter layers
    hooks = []
    for name, module in peft_model.named_modules():
        if 'lora' in name or 'adapter' in name:
            hook = module.register_forward_hook(create_forward_hook(name))
            hooks.append(hook)

    # Perform inference
    predictions, references = [], []
    print("\nStarting inference...")

    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            with autocast():  # Enable mixed precision
                generated_tokens = peft_model.generate(
                    input_features=batch["input_features"].to("cuda"),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=config.get("max_new_tokens", 128),
                    use_cache=True,
                    no_repeat_ngram_size=2,
                    num_beams=5,
                ).cpu().numpy()

            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

            decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(decoded_preds)
            references.extend(decoded_labels)

            # Print sample predictions (first batch only)
            if step == 0:
                for i, (pred, ref) in enumerate(zip(decoded_preds, decoded_labels)):
                    print(f"Sample {i}: Prediction = {pred}, Reference = {ref}")

        # Clean up memory
        del generated_tokens, labels, batch
        gc.collect()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print adapter layer usage
    print("\nAdapter Layer Usage:")
    for name, count in module_usage.items():
        if count > 0:
            print(f"Module {name} was used {count} times during inference.")
        else:
            print(f"Module {name} was not used during inference.")

    # Compute WER
    wer = 100 * metric.compute(predictions=predictions, references=references)
    print(f"\nWER: {wer:.2f}%")


if __name__ == "__main__":
    main()

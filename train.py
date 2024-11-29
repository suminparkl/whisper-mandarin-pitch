import argparse
import yaml
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate

from model_utils import modify_first_conv_layer, SavePeftModelCallback
from data_utils import (
    add_f0_as_additional_channel,
    prepare_dataset,
    DataCollatorSpeechSeq2SeqWithPadding,
)
from data_preparation import prepare_common_voice_dataset


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a Whisper model with additional pitch features"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the configuration file"
    )
    args = parser.parse_args()

    # Load configurations from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set evaluation metrics
    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")  # Additional metric

    # Initialize model and LoRA settings
    model_name_or_path = config["model_name_or_path"]
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # Modify the first Conv1D layer to accept additional pitch channel
    model = modify_first_conv_layer(model)

    # LoRA configuration and application
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q_proj", "v_proj"],
        lora_dropout=config["lora_dropout"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Set training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        num_train_epochs=config["num_train_epochs"],
        evaluation_strategy=config.get("evaluation_strategy", "steps"),
        fp16=config.get("fp16", True),
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        generation_max_length=config["generation_max_length"],
        logging_steps=config["logging_steps"],
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # Define tokenizer and processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name_or_path, language=config["language"], task="transcribe"
    )
    processor = WhisperProcessor.from_pretrained(
        model_name_or_path, language=config["language"], task="transcribe"
    )
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    # Prepare dataset using data_preparation.py
    common_voice = prepare_common_voice_dataset(config)

    # Prepare dataset by adding f0 features and tokenizing labels
    common_voice = common_voice.map(
        lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
        remove_columns=common_voice["train"].column_names,
        num_proc=config.get("num_proc", 4),
        load_from_cache_file=False,
    )

    # Create data collator instance
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback()],
        # Uncomment and implement compute_metrics if needed
        # compute_metrics=lambda p: {
        #     "wer": metric_wer.compute(predictions=p.predictions, references=p.label_ids),
        #     "cer": metric_cer.compute(predictions=p.predictions, references=p.label_ids),
        # }
    )

    # Start training
    model.config.use_cache = False  # Remove cache-related warnings
    trainer.train()


if __name__ == "__main__":
    main()

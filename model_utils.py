import torch
import torch.nn as nn
import os
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


# Function to modify the first Conv1D layer and initialize weights
def modify_first_conv_layer(model, dtype=torch.float16):
    original_conv = model.model.encoder.conv1
    new_conv = nn.Conv1d(
        in_channels=81,  # Increase input channels to include pitch
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
    ).to(dtype=dtype, device="cuda")

    with torch.no_grad():
        new_conv.weight[:, :80, :] = original_conv.weight
        new_conv.weight[:, 80, :] = torch.mean(
            original_conv.weight, dim=1
        )  # Initialize the new channel

    model.model.encoder.conv1 = new_conv
    return model


# Custom callback for saving PEFT model
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

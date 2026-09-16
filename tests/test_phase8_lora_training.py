import yaml

from medreason_agent.data.vqa_rad import VQARADSample
from scripts.train_qwen_vl_lora import (
    TrainingExample,
    build_training_arguments,
    build_training_prompt,
    canonical_training_answer,
)


def _sample() -> VQARADSample:
    return VQARADSample(
        dataset="VQA-RAD",
        sample_id="1",
        image_id="image.jpg",
        image_path="Data/Raw/vqa_rad/images/image.jpg",
        question="Is there pleural effusion?",
        answer="no",
        answer_type="CLOSED",
        question_type="PRES",
        image_organ="CHEST",
        split="train",
    )


def test_training_example_uses_image_question_and_short_answer() -> None:
    example = TrainingExample(_sample())
    user_messages = example.user_messages()
    full_messages = example.full_messages()

    assert user_messages[0]["role"] == "user"
    assert user_messages[0]["content"][0]["type"] == "image"
    assert "Is there pleural effusion?" in user_messages[0]["content"][1]["text"]
    assert full_messages[-1]["role"] == "assistant"
    assert full_messages[-1]["content"][0]["text"] == "no"


def test_training_example_v2_uses_question_type_prompt_and_normalized_target() -> None:
    sample = _sample()
    example = TrainingExample(
        sample,
        prompt_template="question_type_aware_short_v2",
        image_min_pixels=50176,
        image_max_pixels=602112,
    )
    user_messages = example.user_messages()
    prompt = user_messages[0]["content"][1]["text"]
    image_content = user_messages[0]["content"][0]

    assert "Answer type: CLOSED" in prompt
    assert "Question type: PRES" in prompt
    assert "output exactly yes or no" in prompt
    assert image_content["min_pixels"] == 50176
    assert image_content["max_pixels"] == 602112
    assert example.full_messages()[-1]["content"][0]["text"] == "no"


def test_build_training_prompt_v2_includes_modality_constraint() -> None:
    sample = VQARADSample(
        dataset="VQA-RAD",
        sample_id="2",
        image_id="image.jpg",
        image_path="Data/Raw/vqa_rad/images/image.jpg",
        question="What modality is this image?",
        answer="computed tomography",
        answer_type="OPEN",
        question_type="MODALITY",
        image_organ="CHEST",
        split="train",
    )

    prompt = build_training_prompt(sample, "question_type_aware_short_v2")

    assert "xray, ct, mri, ultrasound" in prompt
    assert canonical_training_answer(sample) == "ct"


def test_phase8_training_config_parses() -> None:
    with open(
        "configs/training/phase8_qwen_vl_lora_vqa_rad_4090d.yaml",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    assert config["phase"] == 8
    assert config["method"]["type"] == "qwen_vl_lora_sft"
    assert config["outputs"]["merge_adapter"] is True


def test_phase8_lora_v2_training_config_parses() -> None:
    with open(
        "configs/training/phase8_qwen_vl_lora_v2_format_48gb.yaml",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    assert config["phase"] == 8
    assert config["method"]["prompt_template"] == "question_type_aware_short_v2"
    assert config["vision"]["max_pixels"] == 602112
    assert config["outputs"]["merged_model_dir"].endswith("qwen_vl_lora_v2_format_48gb/merged")


def test_build_training_arguments_handles_old_transformers_signature(tmp_path) -> None:
    class OldTrainingArguments:
        def __init__(
            self,
            output_dir,
            num_train_epochs,
            warmup_steps,
            evaluation_strategy,
            save_strategy,
        ) -> None:
            self.kwargs = {
                "output_dir": output_dir,
                "num_train_epochs": num_train_epochs,
                "warmup_steps": warmup_steps,
                "evaluation_strategy": evaluation_strategy,
                "save_strategy": save_strategy,
            }

    args = build_training_arguments(
        OldTrainingArguments,
        tmp_path,
        {
            "training": {
                "num_train_epochs": 2,
                "warmup_ratio": 0.03,
                "eval_strategy": "steps",
                "save_strategy": "steps",
            }
        },
    )

    assert args.kwargs["warmup_steps"] == 0
    assert args.kwargs["evaluation_strategy"] == "steps"
    assert args.kwargs["save_strategy"] == "steps"

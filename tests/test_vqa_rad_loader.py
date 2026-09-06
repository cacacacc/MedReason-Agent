from medreason_agent.data.vqa_rad import VQARADSample


def test_vqa_rad_sample_from_record() -> None:
    sample = VQARADSample.from_record(
        {
            "dataset": "VQA-RAD",
            "sample_id": "1",
            "image_id": "image.jpg",
            "image_path": "Data/Raw/vqa_rad/image.jpg",
            "question": "Is there opacity?",
            "answer": "yes",
            "answer_type": "CLOSED",
            "question_type": "PRES",
            "image_organ": "CHEST",
            "split": "test",
        }
    )

    assert sample.dataset == "VQA-RAD"
    assert sample.sample_id == "1"
    assert sample.answer_type == "CLOSED"

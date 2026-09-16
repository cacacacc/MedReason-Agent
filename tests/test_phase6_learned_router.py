from medreason_agent.agents.learned_router import (
    DIRECT_ROUTE,
    extract_router_features,
    oracle_route_to_training_label,
    predict_route,
)
from scripts.train_phase6_oracle_router import build_dataset, train_router


def test_oracle_route_to_training_label_maps_none_to_direct() -> None:
    assert oracle_route_to_training_label("none") == DIRECT_ROUTE
    assert oracle_route_to_training_label("structured_cot") == "structured_cot"


def test_extract_router_features_uses_direct_prediction_and_question_markers() -> None:
    features = extract_router_features(
        {
            "question": "Which side has the larger opacity?",
            "answer_type": "OPEN",
            "question_type": "SIZE, PRES",
            "image_organ": "CHEST",
            "direct_prediction": "left side",
            "direct_raw_output": "Final Answer: left side",
        }
    )

    assert features["answer_type"] == "OPEN"
    assert features["question_type_has_SIZE"] is True
    assert features["question_has_which_side"] is True
    assert features["direct_prediction"] == "left"
    assert features["direct_pred_token_left"] is True


def test_train_router_predicts_deployable_route() -> None:
    rows = [
        {
            "question": "Is there pleural effusion?",
            "answer_type": "CLOSED",
            "question_type": "PRES",
            "image_organ": "CHEST",
            "direct_prediction": "no",
            "oracle_route": "direct",
        },
        {
            "question": "Which side has the larger opacity?",
            "answer_type": "OPEN",
            "question_type": "SIZE",
            "image_organ": "CHEST",
            "direct_prediction": "right",
            "oracle_route": "structured_cot",
        },
        {
            "question": "What diagnosis is most consistent?",
            "answer_type": "OPEN",
            "question_type": "OTHER",
            "image_organ": "CHEST",
            "direct_prediction": "pneumonia",
            "oracle_route": "full_multi_agent",
        },
        {
            "question": "Is the image axial?",
            "answer_type": "CLOSED",
            "question_type": "PLANE",
            "image_organ": "HEAD",
            "direct_prediction": "yes",
            "oracle_route": "none",
        },
        {
            "question": "Is there a mass?",
            "answer_type": "CLOSED",
            "question_type": "PRES",
            "image_organ": "ABD",
            "direct_prediction": "no",
            "oracle_route": "direct",
        },
        {
            "question": "How many lesions are present?",
            "answer_type": "OPEN",
            "question_type": "COUNT",
            "image_organ": "HEAD",
            "direct_prediction": "2",
            "oracle_route": "structured_cot",
        },
    ]
    features, labels = build_dataset(rows)
    model, metrics = train_router(
        features=features,
        labels=labels,
        seed=42,
        validation_ratio=0.34,
        class_weight=None,
    )
    prediction = predict_route(model, features[0])

    assert metrics["num_samples"] == 6
    assert prediction.route in {"direct", "structured_cot", "full_multi_agent"}

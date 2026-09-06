from medreason_agent.paths import required_project_paths, resolve_project_path


def test_required_project_paths_exist() -> None:
    missing = [name for name, path in required_project_paths().items() if not path.exists()]
    assert missing == []


def test_base_config_exists() -> None:
    assert resolve_project_path("configs/base.yaml").exists()


def test_initial_experiment_configs_exist() -> None:
    assert resolve_project_path("configs/experiments/exp00_majority.yaml").exists()
    assert resolve_project_path("configs/experiments/exp01_direct_vlm.yaml").exists()


def test_research_docs_exist() -> None:
    docs = [
        "Docs/research_questions.md",
        "Docs/experiment_protocol.md",
        "Docs/error_taxonomy.md",
    ]
    assert all(resolve_project_path(path).exists() for path in docs)

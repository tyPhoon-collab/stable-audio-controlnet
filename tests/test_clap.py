import os
import sys
from pathlib import Path

import pytest

# Change to app directory for relative paths
os.chdir(Path(__file__).parent.parent)
# Add the parent directory to the path to import from main
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.eval.clap import calculate_pair_similarity, initialize_clap_model


@pytest.fixture(scope="session")
def clap_model():
    """Initialize the CLAP model once for all tests."""
    return initialize_clap_model()


def test_calculate_pair_similarity_with_music_description(clap_model):
    """Test similarity calculation with music description and vocal-free audio."""
    description = "An upbeat track featuring acoustic guitars, rhythmic percussion, and bright piano, creating a lively and cheerful mood with a medium tempo."
    audio_path = "data/Al James - Schoolboy Facination_no_vocals.mp3"

    # Calculate similarity
    score = calculate_pair_similarity(clap_model, audio_path, description)
    print(f"\nMusic description score: {score:.4f}")

    # Check that score is a float between -1 and 1
    assert isinstance(score, float), "Score should be a float"
    assert -1 <= score <= 1, f"Similarity score should be between -1 and 1, got {score}"

    # Check that the score is reasonably high for a matching description
    assert score > 0.3, f"Expected score > 0.3 for matching description, got {score}"


def test_calculate_pair_similarity_with_dog_description(clap_model):
    """Test similarity calculation with non-matching description."""
    description = "The dog is barking."
    audio_path = "data/Al James - Schoolboy Facination_no_vocals.mp3"

    # Calculate similarity
    score = calculate_pair_similarity(clap_model, audio_path, description)
    print(f"\nDog description score: {score:.4f}")

    # Check that score is a float between -1 and 1
    assert isinstance(score, float), "Score should be a float"
    assert -1 <= score <= 1, f"Similarity score should be between -1 and 1, got {score}"

    # Check that the score is lower for non-matching description
    # (Not too strict, just check it's not very high)
    assert score < 0.1, (
        f"Expected score < 0.1 for non-matching description, got {score}"
    )


def test_calculate_pair_similarity_consistency(clap_model):
    """Test that the same input produces consistent output."""
    description = "An upbeat track featuring acoustic guitars, rhythmic percussion, and bright piano, creating a lively and cheerful mood with a medium tempo."
    audio_path = "data/Al James - Schoolboy Facination_no_vocals.mp3"

    # Calculate similarity twice
    score1 = calculate_pair_similarity(clap_model, audio_path, description)
    score2 = calculate_pair_similarity(clap_model, audio_path, description)
    print(f"\nConsistency test - Score1: {score1:.4f}, Score2: {score2:.4f}")

    # Scores should be identical
    assert score1 == score2, f"Scores should be identical: {score1} != {score2}"


@pytest.fixture(scope="session")
def clap_model_fusion():
    """Initialize the CLAP model with fusion enabled for all tests."""
    return initialize_clap_model(
        weights_path="ckpts/630k-audioset-fusion-best.pt",
        amodel="HTSAT-tiny",
        enable_fusion=True,
    )


def test_calculate_pair_similarity_with_fusion_model(clap_model_fusion):
    """Test similarity calculation with fusion-enabled model."""
    description = "An upbeat track featuring acoustic guitars, rhythmic percussion, and bright piano, creating a lively and cheerful mood with a medium tempo."
    audio_path = "data/Al James - Schoolboy Facination_no_vocals.mp3"

    # Calculate similarity
    score = calculate_pair_similarity(clap_model_fusion, audio_path, description)
    print(f"\nFusion model score: {score:.4f}")

    # Check that score is a float between -1 and 1
    assert isinstance(score, float), "Score should be a float"
    assert -1 <= score <= 1, f"Similarity score should be between -1 and 1, got {score}"

    # Check that the score is reasonably high for a matching description
    assert score > 0.0, f"Expected score > 0.0 for matching description, got {score}"


def test_calculate_pair_similarity_fusion_with_dog_description(clap_model_fusion):
    """Test similarity calculation with fusion model and non-matching description."""
    description = "The dog is barking."
    audio_path = "data/Al James - Schoolboy Facination_no_vocals.mp3"

    # Calculate similarity
    score = calculate_pair_similarity(clap_model_fusion, audio_path, description)
    print(f"\nFusion model dog description score: {score:.4f}")

    # Check that score is a float between -1 and 1
    assert isinstance(score, float), "Score should be a float"
    assert -1 <= score <= 1, f"Similarity score should be between -1 and 1, got {score}"

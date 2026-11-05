from src.utils.progress import (
    print_key_values,
    short_metric_name,
    training_progress,
    wrapped_text,
)


def test_wrapped_text_respects_requested_width():
    message = (
        "Epoch 12/50000: training_loss=1.234567e-03, "
        "normalized_pde=8.765432e-04, normalized_jacobian=2.345678e-04, "
        "singular_value=1.234567e-04, mse=9.876543e-05"
    )
    rendered = wrapped_text(message, width=72)
    assert len(rendered.splitlines()) > 1
    assert max(map(len, rendered.splitlines())) <= 72
    assert "training_loss=1.234567e-03" in rendered


def test_print_key_values_wraps_between_metrics(capsys):
    print_key_values(
        "Epoch 3/100: ",
        [
            ("training_loss", 1.2e-3),
            ("pde_loss", 2.3e-4),
            ("jacobian_loss", 3.4e-5),
            ("dataset_mse", 4.5e-6),
        ],
        width=68,
    )
    output = capsys.readouterr().out.rstrip()
    assert len(output.splitlines()) > 1
    assert max(map(len, output.splitlines())) <= 68
    assert "dataset_mse=4.500000e-06" in output


def test_training_progress_keeps_all_losses_with_short_names():
    progress = training_progress(range(1), disable=True, desc="Epoch 1/50000")
    progress.set_postfix(
        loss=1.0,
        pde_loss=2.0,
        fixed_mse=3.0,
        jacobian_loss=4.0,
        singular_value_loss=5.0,
    )
    try:
        assert progress.postfix.count("=") == 5
        assert "loss=" in progress.postfix
        assert "pde=" in progress.postfix
        assert "jac=" in progress.postfix
        assert "sv=" in progress.postfix
        assert "pde_loss=" not in progress.postfix
    finally:
        progress.close()


def test_short_metric_name_preserves_persisted_names_only_in_storage():
    assert short_metric_name("normalized_jacobian") == "jac"
    assert short_metric_name("singular_value") == "sv"
    assert short_metric_name("custom_metric") == "custom_metric"

import torch
from analysis.example_selection import collect_conditional_examples
from analysis.attention_measurement import measure_attention_on_examples

def measure_conditional_attention_statistics(
    model,
    loader,
    device,
    conditional_tokens,
    num_samples=500,
    max_batches=1000,
    previous_tokens=None,
):
    examples = collect_conditional_examples(
        loader=loader,
        device=device,
        conditional_tokens=conditional_tokens,
        num_samples=num_samples,
        max_batches=max_batches,
        previous_tokens=previous_tokens,
    )

    stats = measure_attention_on_examples(
        model=model,
        loader=loader,
        device=device,
        example_indices=examples["indices"],
    )

    return stats, examples

def get_control_statistics(examples, model, loader, device, conditional_tokens):
    """
    given collection of conditional examples, compute attention statistics on 
    conditional_tokens with second to last token fixed to match statistics of
    prev_token in examples
    """
    prev_controls, counts = examples['prev_token'].unique(return_counts=True)

    control_indices = []

    for tok, cnt in zip(prev_controls, counts):
        ctrl_examples = collect_conditional_examples(
            loader,
            device,
            conditional_tokens=conditional_tokens,
            previous_tokens=[tok.item()],
            num_samples=cnt.item()
        )
        control_indices.append(ctrl_examples["indices"])

    control_indices = torch.cat(control_indices)

    control_stats = measure_attention_on_examples(
        model, loader, device, control_indices
    )

    return control_stats
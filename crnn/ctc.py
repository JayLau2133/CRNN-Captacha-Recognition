from .config import BLANK_IDX, CHAR_SET


def ctc_greedy_decode(predictions, input_lengths=None):
    if predictions.ndim != 3:
        raise ValueError("Expected predictions with shape [time, batch, classes]")
    indices = predictions.argmax(dim=-1)
    if input_lengths is None:
        input_lengths = [indices.size(0)] * indices.size(1)

    decoded = []
    for batch_index, length in enumerate(input_lengths):
        characters = []
        previous_index = -1
        for index in indices[:int(length), batch_index].tolist():
            if index != BLANK_IDX and index != previous_index:
                characters.append(CHAR_SET[index])
            previous_index = index
        decoded.append("".join(characters))
    return decoded


def edit_distance(source, target):
    previous = list(range(len(target) + 1))
    for source_index, source_character in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_character in enumerate(target, start=1):
            current.append(min(
                current[-1] + 1,
                previous[target_index] + 1,
                previous[target_index - 1] + (source_character != target_character),
            ))
        previous = current
    return previous[-1]


def calculate_metrics(predicted_strings, true_strings):
    if len(predicted_strings) != len(true_strings):
        raise ValueError("Predictions and targets must have the same length")
    correct = sum(
        prediction == target
        for prediction, target in zip(predicted_strings, true_strings)
    )
    errors = sum(
        edit_distance(prediction, target)
        for prediction, target in zip(predicted_strings, true_strings)
    )
    character_count = sum(len(target) for target in true_strings)
    if character_count == 0:
        raise ValueError("CER is undefined for empty target strings")
    return {
        "accuracy": correct / len(true_strings) if true_strings else 0.0,
        "cer": errors / character_count,
        "samples": len(true_strings),
    }

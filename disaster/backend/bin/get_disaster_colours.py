from pathlib import Path
import json

BACKGROUND_ALPHA = 50


if __name__ == "__main__":

    samples = []
    for label_path in Path("../../../data/input/train/labels").glob("*.json"):
        with label_path.open() as f:
            samples.append(json.load(f))

    scenarios = list(
        set(
            [(sample["metadata"]["disaster"], sample["metadata"]["disaster_type"])
             for sample in samples]
        )
    )
    # Sort by the disaster type so that disasters of the same
    # type have similar colours
    scenarios.sort(key=lambda sample: sample[1])

    chunk_size = round((255 * 3) / len(scenarios))

    current_chunk = 0
    background_colour_mapping = {}
    for scenario in scenarios:

        background_colour_mapping[scenario[0]] = (
            # Red
            max(0, min(current_chunk, 255)),
            # Green
            max(0, min(current_chunk - 255, 255)),
            # Blue
            max(0, min(255, current_chunk - (255 * 2))),
            # Alpha
            BACKGROUND_ALPHA
        )
        current_chunk += chunk_size

    print(background_colour_mapping)

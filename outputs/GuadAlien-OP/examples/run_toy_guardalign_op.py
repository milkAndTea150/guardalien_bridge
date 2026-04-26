from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.guardalign_op import guardalign_op_score


def main() -> None:
    torch.manual_seed(7)

    image_embeds = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    text_embeds = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    result = guardalign_op_score(image_embeds, text_embeds, epsilon=0.05, num_iters=200)

    print("Cost matrix:")
    print(result["cost_matrix"])
    print("\nTransport plan:")
    print(result["transport_plan"])
    print("\nPatch suspiciousness scores:")
    print(result["patch_scores"])


if __name__ == "__main__":
    main()

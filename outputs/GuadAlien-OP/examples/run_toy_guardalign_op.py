import torch

from src.guardalign_op import guardalign_op_score


def main():
    torch.manual_seed(0)
    image_embeds = torch.randn(4, 8)
    unsafe_text_embeds = torch.randn(3, 8)
    result = guardalign_op_score(image_embeds, unsafe_text_embeds)

    print("cost_matrix shape:", tuple(result["cost_matrix"].shape))
    print("transport_plan shape:", tuple(result["transport_plan"].shape))
    print("patch_scores:", result["patch_scores"].detach().cpu().tolist())
    print("global_score:", float(result["global_score"]))


if __name__ == "__main__":
    main()

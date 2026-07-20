import torch
import torch.nn.functional as F

def logits_to_grade(logits: torch.Tensor, strategy: str = "argmax") -> int:
    probs = F.softmax(logits, dim=1).squeeze()
    if strategy == "argmax":
        return int(probs.argmax().item())
    elif strategy == "expected_value":
        grades = torch.arange(len(probs), dtype=torch.float, device=logits.device)
        return int(round(torch.dot(probs, grades).item()))
    raise ValueError(f"Unknown strategy: {strategy}")

def softmax_to_grade(probabilities: list[float], strategy: str = "argmax") -> int:
    probs_tensor = torch.tensor(probabilities).unsqueeze(0)
    return logits_to_grade(probs_tensor, strategy)

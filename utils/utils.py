def check_grad(model):
    with open("output.txt", "w") as file:
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"After backward, {name} has gradient: {param.grad}", file=file)
            else:
                print(f"After backward, {name} has no gradient", file=file)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())




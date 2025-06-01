from klay import Circuit
import torch

def main():
    print("KLay Query")

    # 1. Instantiate Circuit and load the SDD (vtree loaded automatically)
    circ = Circuit()
    # Only pass the SDD filename; KLay will look for the same prefix + '.vtree'
    root = circ.add_sdd_from_file('sdd_model.sdd', [])

    # 2. Clean up any unused nodes
    circ.remove_unused_nodes()

    torch_module = circ.to_torch_module()

    # 4. Switch to eval mode and run without gradients
    torch_module.eval()
    with torch.no_grad():
        output = torch_module()  # Tensor of shape (num_roots,)

    # 5. Extract and print the probability for our single root
    prob = output[0].item()
    print(f"P(conflict_intensity(france, germany)) = {prob:.6f}")


if __name__ == '__main__':
    main()


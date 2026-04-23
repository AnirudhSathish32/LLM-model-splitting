from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./llama-3b"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

stopping_layer = 14
starting_layer = 15

inputs = tokenizer("Hello world", return_tensors="pt")

def capture_full_pass():
    """
    Full forward pass through all 28 layers.
    Captures layer 14 output and layer 15 output using hooks.
    This is the ground truth we compare everything else against.
    """
    captured = {}

    def hook_layer_stopping(module, input, output):
        captured["stopping"] = output[0].detach().clone()
        if output[0].dim() == 2:
            captured["stopping"] = captured["stopping"].unsqueeze(0)        

    def hook_layer_starting(module, input, output):
        captured["starting"] = output[0].detach().clone()
        if output[0].dim() == 2:
            captured["starting"] = captured["starting"].unsqueeze(0) 

    h1 = model.model.layers[stopping_layer - 1].register_forward_hook(hook_layer_stopping)
    h2 = model.model.layers[starting_layer - 1].register_forward_hook(hook_layer_starting)
    print(len(model.model.layers))
    with torch.no_grad():
        model(**inputs)

    h1.remove()
    h2.remove()
    
    print("====== stopping layer ======")
    print(captured["stopping"])

    print("====== starting layer ======")
    print(captured["starting"])

    return captured["stopping"], captured["starting"]


def capture_stopped_pass():
    """
    Runs the model and stops immediately after layer 13 finishes.
    Layers 14-27 never execute.
    Returns the layer 14 output and the position info needed to resume.
    """
    captured = {}

    def hook_fn(module, input, output):
        hidden = output[0].detach().clone()
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        captured["stopping"] = hidden
        raise StopIteration

    def hook_pos(module, args, kwargs):
        cos, sin = kwargs.get("position_embeddings")
        captured["position_embeddings"] = (cos.detach().clone(), sin.detach().clone())
        captured["position_ids"] = kwargs.get("position_ids")

    h2 = model.model.layers[stopping_layer - 1].register_forward_pre_hook(hook_pos, with_kwargs=True)
    h1 = model.model.layers[stopping_layer - 1].register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            model(**inputs)
    except StopIteration:
        pass

    h1.remove()
    h2.remove()

    return captured["stopping"], captured["position_ids"], captured["position_embeddings"]



def capture_partial_pass(layer14_output, position_ids, position_embeddings):
    with torch.no_grad():
        x = layer14_output
        x = model.model.layers[starting_layer - 1](
            x,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )[0]
        if x.dim() == 2:
            x = x.unsqueeze(0)
    return x

if __name__ == "__main__":
    full_stopping_layer, full_starting_layer = capture_full_pass()
    stopped_stop_layer, pos_ids, pos_emb = capture_stopped_pass()
    partial_start_layer = capture_partial_pass(stopped_stop_layer, pos_ids, pos_emb)

    print("Stopping Layer match:", torch.allclose(full_stopping_layer, stopped_stop_layer, atol=1e-2))
    print("Starting Layer match:", torch.allclose(full_starting_layer, partial_start_layer, atol=1e-2))
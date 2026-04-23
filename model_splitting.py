"""
VALIDATION STRATEGY: Hidden State Equivalence
--------------------------------------------------------------------------------
Why we only compare output[0] (the 'Hidden State'):

In a Transformer model, data flows through layers like a relay race. 
The 'Hidden State' is the 'baton' a 3D tensor containing the model's 
compressed conceptual understanding of the input at that specific layer.

1. The 'Source of Truth':
   Each layer in the Transformer is mathematically defined to receive 
   the hidden state from the previous layer as its primary input. 
   If our split is successful, the input to Layer 15 (our 'partial pass') 
   must be identical to the input Layer 15 would have received in a 
   standard, continuous execution.

2. Defining Equivalence:
   Because the model is a deterministic chain of mathematical operations, 
   the 'Hidden State' is the sole driver of the next layer's output. If the 
   Hidden State at Layer 14 is identical (within floating-point tolerance), 
   then the output of Layer 15, 16, and all subsequent layers is guaranteed 
   to be identical, regardless of whether the model was run in one piece 
   or split into chunks.

3. Ignoring Auxiliary Data:
   While each layer returns a tuple (a package containing hidden states, 
   attention weights, and cache objects), only the hidden state (index 0) 
   feeds into the next layer's 'forward' function. The auxiliary data acts 
   as 'spectator' information for this specific workflow. Comparing the 
   Hidden State alone is sufficient to prove that our model-splitting logic 
   maintains the mathematical integrity of the entire inference pipeline.
--------------------------------------------------------------------------------
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./llama-3b"

model = AutoModelForCausalLM.from_pretrained(model_path)
#Loading model 

tokenizer = AutoTokenizer.from_pretrained(model_path)
#Loading Tokenizer

prompt = "Hello world"
#Loading Prompt

inputs = tokenizer(prompt, return_tensors="pt")
#Tokenizing prompt into input tensors

model.eval()

"""
Defining Split Boundary: 

Stop Layer defines the end of the first split
Start Layer defines the beginning of the second split

"""

stopping_layer = 14
#Defining Stop Layer 

starting_layer = 15
#Defining start layer



def capture_full_pass():
    """
    Full forward pass through all 28 layers.
    Captures layer 14 output and layer 15 output using hooks.
    This is the ground truth we compare everything else against.
    """
    captured = {}
    #Empty Dictionary to store stopping and starting layer outputs

    def hook_layer_stopping(module, input, output):
        """
        Forward hook to capture output of stopping layer

        What is a forward hook?
            A user-defined function that allows us to "register" a layer
            PyTorch is instructed to call this function every time a layer calls the forward() method

        Two types of forward hooks 
            Forward pre-hook: Executes before the layer does its math we can see the input data
            
            Forward hook: Executes after the layer completes its math but before the data is 
            passed into the next layer
        
        Args:
            module: The PyTorch layer the hook is attached to.
            input: The input tuple to the layer.
            output: The output tuple returned by the layer.

        Details:
        - Intercepts the forward pass at the specified layer.
        - Extracts the primary hidden state (index 0).
        - Calls .detach() to disconnect from the gradient computation graph.
        - Calls .clone() to create an independent memory copy of the tensor.

        output[0] is the hidden state (multidimensional tensors) calculated by the current layer
        .detach() frees up memory from PyTorch. Removes the history tracking for each tensor 
        .clone() Ensures the data we save doesnt get overwritten by in place operations 
        """

        captured["stopping"] = output[0].detach().clone()
        # Captures hidden state data

        if output[0].dim() == 2:
            captured["stopping"] = captured["stopping"].unsqueeze(0)  
        
        # Fixes dimensions of the output 

    def hook_layer_starting(module, input, output):
        """
        Forward hook to capture output of stopping layer

        What is a forward hook?
            A user-defined function that allows us to "register" a layer
            PyTorch is instructed to call this function every time a layer calls the forward() method

        Two types of forward hooks 
            Forward pre-hook: Executes before the layer does its math we can see the input data
            
            Forward hook: Executes after the layer completes its math but before the data is 
            passed into the next layer
        
        Args:
            module: The PyTorch layer the hook is attached to.
            input: The input tuple to the layer.
            output: The output tuple returned by the layer.

        Details:
        - Intercepts the forward pass at the specified layer.
        - Extracts the primary hidden state (index 0).
        - Calls .detach() to disconnect from the gradient computation graph.
        - Calls .clone() to create an independent memory copy of the tensor.

        output[0] is the hidden state (multidimensional tensors) calculated by the current layer
        .detach() frees up memory from PyTorch. Removes the history tracking for each tensor 
        .clone() Ensures the data we save doesnt get overwritten by in place operations 
        """

        captured["starting"] = output[0].detach().clone()
        # Captures hidden state data 

        if output[0].dim() == 2:
            captured["starting"] = captured["starting"].unsqueeze(0) 
        # Fixes dimensions of the output 
        # Input of the next layer wants a 3 dimensional shape but hook returns a 2D shape

        print(captured["starting"].shape)

    h1 = model.model.layers[stopping_layer - 1].register_forward_hook(hook_layer_stopping)
    # We call model.model.layer[x] to access a specific model layer
    # We call register_forward_hook to attach the hook we defined to a specific layer
    # Stopping_layer - 1 because of 0 indexing 

    h2 = model.model.layers[starting_layer - 1].register_forward_hook(hook_layer_starting)
    # We call model.model.layer[x] to access a specific model layer
    # we call register_forward_hook to attach the hook we defined to a specific layer
    # Starting_layer - 1 because of 0 indexing

    #print(len(model.model.layers))

    with torch.no_grad():
        model(**inputs)
    # Starting the forward pass 
    # Denoted with **to turn the input into a dictionary where every key is an 
    # argument name  

    h1.remove()
    # removing hook 1
    h2.remove()
    # removing hook 2
    
    #print("====== stopping layer ======")
    #print(captured["stopping"])

    #print("====== starting layer ======")
    #print(captured["starting"])

    return captured["stopping"], captured["starting"]


def capture_stopped_pass():
    """
    Runs the model and stops immediately after layer 14 finishes.
    Layers 14-28 never execute.
    Returns the layer 14 output and the position info needed to resume.

    Position ids and position embeddings are static constants generated at layer 0 
    and carried through each layer

    If we want to create a valid input for layer 15 we need the layer 0 information

    
    

    """
    captured = {}

    def hook_fn(module, input, output):
        hidden = output[0].detach().clone()
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        captured["stopping"] = hidden
        raise StopIteration
        # Stops iteration when this hook is launched

    def hook_pos(module, args, kwargs):
        cos, sin = kwargs.get("position_embeddings")
        # Llama-3B defines these position embeddings as cos, sin and performs RoPE - Rotary Position Embeddings
        captured["position_embeddings"] = (cos.detach().clone(), sin.detach().clone())
        # Same .detach .clone logic as seen before
        captured["position_ids"] = kwargs.get("position_ids")
        

    h2 = model.model.layers[stopping_layer - 1].register_forward_pre_hook(hook_pos, with_kwargs=True)
    # position ids and embeddings are stored in the arguments for the input of each layer sp we need forward prehook to access
    h1 = model.model.layers[stopping_layer - 1].register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            model(**inputs)
    except StopIteration:
        pass
    # gracefully handles the stopping

    h1.remove()
    h2.remove()

    return captured["stopping"], captured["position_ids"], captured["position_embeddings"]



def capture_partial_pass(stop_layer_output, position_ids, position_embeddings):
    """
    Performs forward pass only on the starting layer to get its output
    We use to confirm if the input package works
    """
    
    with torch.no_grad():
        for i in range(starting_layer - 1, starting_layer):
            x = model.model.layers[i](
                stop_layer_output,
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

    
    print("Stopping Layer match:", torch.allclose(full_stopping_layer, stopped_stop_layer))
    # torch.allclose is a boolean method which checks to see if two tensors are


    print("Starting Layer match:", torch.allclose(full_starting_layer, partial_start_layer))
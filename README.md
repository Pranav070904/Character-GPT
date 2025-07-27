# Character-Level GPT in PyTorch

This project is a from-scratch implementation of a decoder-only transformer model (similar to GPT) for character-level text generation. The model is built using PyTorch and trained on the "tiny Shakespeare" dataset to generate text in the style of Shakespeare.

This code is a self-contained script that handles data preprocessing, model definition, training, and text generation. It's designed to be a clear and educational example of how modern transformer architectures work.

## Features

- Decoder-Only Architecture: Implements a GPT-style model focused on text generation.
- Multi-Head Self-Attention: The core of the transformer, allowing the model to weigh the importance of different characters in the context.
- Positional Embeddings: Enables the model to understand the order of characters in a sequence.
- Feed-Forward Networks: A key component of each transformer block.
- Residual Connections & Layer Normalization: Stabilizes training and improves performance, using a pre-norm formulation (LayerNorm -> Attention/FFN -> Add).
- Dropout: Used for regularization to prevent overfitting.
- Autoregressive Generation: Generates new text one character at a time based on the preceding context.


## Model Architecture

The model is constructed from a stack of identical Transformer Blocks. Each block consists of two main sub-layers:

1. Masked Multi-Head Self-Attention: This layer allows each position in the input sequence to attend to all previous positions (but not future ones, thanks to the causal mask). This is what allows the model to learn relationships between characters.

2. Feed-Forward Network (FFN): A simple fully connected neural network that processes the output of the attention layer, adding computational depth.

Residual connections and layer normalization are applied around each of these two sub-layers to ensure stable and effective training.

```bash
    Input -> Embedding -> + Positional Encoding ->[Transformer Block] x N -> Final LayerNorm->Linear Layer-> Output Logits
```

## Code Breakdown

- **Data Loading and Preporcessing:** The script loads in the input.txt file and creates a character level vocabulary . It then defines an encode and decode function to map characters to integer tokens. The data is then split into training and validation sets.
- ```get_batch(split)``` A function that generates a random batch of input sequences ```(x)``` and corresponding target sequences ```(y)``` from either the training or validation data.

- ```Head``` **class:** A single head of self attention

- ```MultiHeadAttention``` **class:** Combines multiple self attention heads in parallel and concatenates their results.

- ```FeedForward``` **class:** A simple multilayer perceptrons.

-```Block``` **class:** A single transformer block, containing a ```MuliHeadAttention``` layer, a ```FeedForward``` layers, LayerNorms and Residual Connections.

- ```BigramLangModel``` **class:** The main model class that ties everything together.

    - ```__init__```**:** Defines the layers: token/positional embeddings, a sequence of Blocks, and a final linear layer to produce logits.

    - ```forward```**:** Defines the forward pass, calculating the loss if targets are provided.

    - ```generate```**:** Performs autoregressive text generation by repeatedly feeding the model's own output back to itself.

- **Training Loop:** The main training loop, initializes the optimizer and runs the training steps.
 
- ```estimate_loss()```**:** a helper function that evaluates model performance at regular intervals.


## Hyperparameters

The key hyperparameters are defined at the top of the script and can be easily modified for experimentation:

- ```block_size = 256```: The context length (how many characters the model looks at).

- ```batch_size = 64```: Number of sequences processed in parallel.

- ```epochs = 5000```: Total number of training iterations.

- ```lr = 3e-4```: Learning rate for the AdamW optimizer.

- ```n_embed = 384```: The dimensionality of the token and positional embeddings.

- ```n_head = 6```: The number of attention heads.

- ```n_layer = 6```: The number of transformer blocks stacked together.

- ```dropout = 0.2```: The dropout rate for regularization.


## Acknowledgements

This project is heavily inspired by Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out." video tutorial. The goal of this project was to learn about the workings of LLMs and transformers under the hood and his work is an invaluable resource for understanding these models from the ground up.
import math
import numpy as np
import matplotlib.pyplot as plt


def softmax(values):
    """Numerically stable softmax across each row."""
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def self_attention(x, wq, wk, wv):
    q = x @ wq
    k = x @ wk
    v = x @ wv
    scores = (q @ k.T) / math.sqrt(q.shape[-1])
    weights = softmax(scores)
    output = weights @ v
    return output, weights


def multi_head_attention(x, head_weights, output_projection):
    """Run two independent attention heads, concatenate, then project."""
    head_outputs = []
    head_attention = []
    for wq, wk, wv in head_weights:
        output, weights = self_attention(x, wq, wk, wv)
        head_outputs.append(output)
        head_attention.append(weights)
    combined = np.concatenate(head_outputs, axis=1)
    final_output = combined @ output_projection
    return final_output, head_attention


def sinusoidal_positional_encoding(positions=10, dimensions=8):
    encoding = np.zeros((positions, dimensions))
    for pos in range(positions):
        for i in range(dimensions):
            angle = pos / (10000 ** ((2 * (i // 2)) / dimensions))
            if i % 2 == 0:
                encoding[pos, i] = math.sin(angle)
            else:
                encoding[pos, i] = math.cos(angle)
    return encoding


def huggingface_examples():
    """Optional demos; downloads small models when run for the first time."""
    try:
        from transformers import pipeline
        fill_mask = pipeline("fill-mask", model="google-bert/bert-base-uncased")
        print("\nBERT fill-mask example:")
        print(fill_mask("The capital of France is [MASK].")[0])


        text_generator = pipeline("text-generation", model="openai-community/gpt2")
        print("\nGPT-2 text-generation example:")
        print(text_generator("Transformers are useful in natural language processing because", max_new_tokens=25,
                             do_sample=False, pad_token_id=50256)[0]["generated_text"])
    except Exception as error:
        print("\nHugging Face examples could not run.")
        print("Check internet access and installed torch/transformers packages.")
        print("Error:", error)


def main():
    np.random.seed(7)
    # Four tokens represented by four 4-dimensional vectors.
    x = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 2.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 2.0, 1.0],
    ])


    # Single-head attention, dimensions remain 4.
    wq = np.random.rand(4, 4)
    wk = np.random.rand(4, 4)
    wv = np.random.rand(4, 4)
    output, weights = self_attention(x, wq, wk, wv)
    print("Single-head attention weights (4x4):")
    print(np.round(weights, 4))
    print("Row sums:", np.round(weights.sum(axis=1), 6))
    assert np.allclose(weights.sum(axis=1), 1.0)
    print("Every attention row sums to 1.0:", np.allclose(weights.sum(axis=1), 1.0))


    # Two heads, each with 2-dimensional Q/K/V outputs. Concatenation gives 4 dims.
    head_weights = []
    for _ in range(2):
        head_weights.append((
            np.random.rand(4, 2),
            np.random.rand(4, 2),
            np.random.rand(4, 2),
        ))
    output_projection = np.random.rand(4, 4)
    multi_output, head_matrices = multi_head_attention(x, head_weights, output_projection)
    print("\nTwo-head output shape:", multi_output.shape)
    print("Head 1 attention weights:\n", np.round(head_matrices[0], 3))
    print("Head 2 attention weights:\n", np.round(head_matrices[1], 3))
    print("Projected output:\n", np.round(multi_output, 3))


    encoding = sinusoidal_positional_encoding(10, 8)
    print("\nSinusoidal positional encoding (10 positions x 8 dimensions):")
    print(np.round(encoding, 3))
    plt.figure(figsize=(8, 4))
    plt.imshow(encoding, aspect="auto")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.title("Sinusoidal Positional Encoding")
    plt.colorbar(label="Encoding value")
    plt.tight_layout()
    plt.savefig("Week_4_Assessment/png/positional_encoding.png")
    print("Saved plot to positional_encoding.png")
    plt.show()


    print("\nEncoder-only vs decoder-only (6 lines):")
    print("1. Encoder-only models such as BERT read input in both directions.")
    print("2. They are useful for understanding text, classification, and masked-token prediction.")
    print("3. Decoder-only models such as GPT predict the next token from earlier tokens.")
    print("4. They are widely used for text generation and instruction following.")
    print("5. Transformers process token relationships with attention and can parallelize training.")
    print("6. This often handles long-range dependencies better and trains more efficiently than sequential RNNs.")


    huggingface_examples()


if __name__ == "__main__":
    main()
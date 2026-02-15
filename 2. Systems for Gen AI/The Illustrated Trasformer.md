# <Paper Title>The Illustrated Transformer for "Attention is All You Need"

*Jay Alammar (Cohere/Writer), Ashish Vaswani (Google Brain), Noam Shazeer (Google Brain), Niki Parmar (Google Research), Jakob Uszkoreit (Google Research), Llion Jones (Google Research), Aidan N. Gomez (University of Toronto), Łukasz Kaiser (Google Brain), Illia Polosukhin.*

**Paper Link:**  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
**Visualization Link:** [Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
**Code / Repo:** N.A  

[PDF](../pdfs/Systems1.pdf)


---

## TL;DR
The Transformer architecture eliminates the need for recurrence (RNNs) in sequence processing, replacing it entirely with an Attention mechanism. By processing all words in a sequence simultaneously rather than one-by-one, it enables massive parallelization and superior handling of long-range dependencies. This post breaks down the complex math of Self-Attention, Multi-Head Attention, and Positional Encodings into intuitive visual blocks, explaining how the model "attends" to different words to build a context-rich representation.

---

## Overview
The paper addresses the fundamental bottleneck of sequential processing in models like RNNs and LSTMs, which are slow to train and struggle to remember information from the beginning of long sentences. The Transformer's high-level approach is to treat a sequence as a set of signals that can be processed in parallel using "Attention," which calculates the relevance of every word to every other word in the sequence. A key assumption is that the "position" of a word can be represented as a signal added to its data, rather than being inherent to the processing order.

---

## System / Model Abstraction
The model uses an Encoder-Decoder structure.

+ Entities: A stack of six Encoders and six Decoders (though the number is arbitrary). Each layer contains a Self-Attention mechanism and a Feed-Forward Neural Network.

+ Interactions: Each word in the input flows through its own path in the encoder, but paths interact during the Self-Attention stage to share context.

+ Guarantees: The architecture enforces "permutation equivariance" (treating input as a set) unless Positional Encodings are added to restore ordering.

---

## System / Model Flow
1. Input Embedding: Words are converted into vectors.

2. Positional Encoding: A specific signal is added to each vector so the model knows the word's position.

3. Encoder Stack:
    + Self-Attention: For each word, the model creates Query (Q), Key (K), and Value (V) vectors.
    + Scoring: It dot-products the current word’s Q with all other words’ K to determine "focus".
    + Weighting: It scales the scores, applies Softmax, and multiplies by the V vectors to get a weighted sum.

4. Decoder Stack:
    + Masked Self-Attention: Similar to the encoder but prevents the model from "cheating" by looking at future words.
    + Encoder-Decoder Attention: Helps the decoder focus on relevant parts of the input sequence.

5. Output: A Linear layer and Softmax turn the vectors into word probabilities.

---

## Math / Important Diagrams

1. The Attention Equation: $$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$Note: $QK^T$ is the "Score," Softmax is the "Normalized Weight," and $V$ is the "Information content."'

2. Multi-Head Attention:$$MultiHead(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O$$ Where each head uses different $W^Q, W^K, W^V$ weight matrices.


---

## Current State of the Art
At the time of the paper, the state-of-the-art was dominated by Recurrent Neural Networks (RNNs), LSTMs, and GRUs. These relied on a "sequential" assumption—processing word t required the hidden state from word t-1. The Transformer challenged this by proving that recurrence is not necessary for high-quality sequence modeling and is actually a hindrance to scaling.

---

## Key Contributions

The main contribution is the Self-Attention mechanism, which allows for global dependencies to be captured regardless of distance in the sequence.

*CLEVER*: Multi-Head Attention: Instead of one attention "view," the model uses multiple "heads" to focus on different types of relationships (e.g., one for grammar, one for pronouns) simultaneously.

Residual Connections & Layer Norm: Used in every sub-layer to stabilize the training of deep stacks.

*CLEVER*: Positional Encodings: Using sine and cosine functions of different frequencies to inject order without using recurrence.

---

## Analogies & Intuitive Explanation

The Library Analogy: Imagine you are writing a research paper. Each word in your sentence is a "Query." You go to a library where every book has a "Key" (the title/topic) and "Value" (the actual content). You compare your Query to all Keys, and the most relevant Values are what you "attend" to and include in your notes.

The Spotlight: RNNs are like a person walking through a dark room with a candle; they only see what is right in front of them and slowly forget what was behind. A Transformer is like turning on the overhead lights; you can see every object in the room at once and instantly see how the chair relates to the table across the room.

---

## AI vs Systems Boundary

Learned: The weights of the $W^Q, W^K, W^V$ matrices, the Feed-Forward Network parameters, and the output Linear layer.

Engineered: The number of heads and layers (hyperparameters), the fixed Positional Encoding formula, and the specific architecture of residual connections. The engineered components are critical to ensuring the model can actually converge during training.

---

## Potential Impact
This work laid the foundation for virtually all modern Large Language Models (LLMs), including BERT, GPT-2, and GPT-3. For practitioners, it shifted the focus from engineering complex recurrent cells to scaling "Attention" blocks on GPUs, leading to a massive leap in translation, summarization, and chat capabilities.

---

## Assumptions

- Explicit: The sequence order matters and can be captured via positional signals.

- Implicit: There is enough GPU memory to handle the $O(n^2)$ complexity of the attention matrix where $n$ is sequence length. It also assumes that "meaning" is primarily contextual—a word is defined by its neighbors.
---


## Risks, Failure Modes, Limitations
 + Quadratic Scaling: Because every word looks at every other word, doubling the sentence length quadruples the memory requirement.
    
+ Loss of Absolute Order: While positional encodings help, the model is inherently "bag-of-words" and can sometimes struggle with strict structural ordering compared to RNNs.

+ Data Hunger: Transformers generally require much larger datasets to beat RNNs because they lack the "inductive bias" of sequential processing.


---

## Costs & Adoption Barriers
 + Compute: Training requires significant GPU clusters due to the parallel nature and high parameter counts.

+ Implementation: While conceptually elegant, the "tensor shuffling" involved in multi-head attention can be non-intuitive to debug for engineers.

+ Long Context: Until recent optimizations (like FlashAttention), processing very long documents remained prohibitively expensive.
---

## Special Notes

+ Prompts: None found.

---


## FAQs & Discussion

**1. What is Positional Encoding and how big is it?**

The Positional Encoding (PE) is a vector of the exact same size as the word embedding ($d_{model}$). In the original Transformer, this is 512. It is added directly to the embedding ($Embedding + PE$). It uses sine and cosine functions of different frequencies to ensure every position has a unique signature that the model can use to determine the distance between tokens.

**2. Are there other types of encodings?**

Yes. Beyond the fixed sinusoidal version, models like BERT use Learned Positional Embeddings (treating positions like a vocabulary). Modern models (Llama, GPT-4) often use RoPE (Rotary Positional Embeddings), which rotates the vectors in a way that naturally captures relative distance, allowing the model to handle longer sequences than it was trained on.


**3. How is Self-Attention different from Masked Attention?**

In an Encoder (and ViTs), we use Self-Attention, where every token can "see" every other token. In a Decoder (used for generation), we use Masked Self-Attention. Since the model generates text one word at a time, we must block it from "looking into the future." We do this by setting the attention scores of future tokens to $-\infty$ before the Softmax step, effectively making them invisible.

**4. What is the "KV Magic" (Weighted Sum)?**

Think of it as a retrieval system:

+ Query (Q): "What am I looking for?"
+ Key (K): "What do I contain?"
+ Value (V): "Here is the actual information."

The dot product $Q \cdot K^T$ calculates a score. After Softmax, these scores become weights. The final output is a weighted average of all the Values. You are essentially filtering the information in the sequence based on relevance to your current word.

**5. Deep Dive: Vanishing Gradients and Normalization**

+ Vanishing Gradient: In deep networks, we use the "Chain Rule" to train. We multiply gradients layer by layer. If those gradients are even slightly less than 1 (e.g., 0.1), multiplying them 50 times results in a number so tiny ($0.1^{50}$) that it "vanishes" to zero. The early layers then stop learning. Residual Connections (the "Add" in Add & Norm) create a shortcut: $Output = x + Layer(x)$. Because of the "$x +$" part, the gradient has a "highway" to travel back without getting shrunk, ensuring it stays at least 1.+1

+ Activations & Normalization: "Activations" are simply the output numbers of a layer (the result of the math). If some activations are 1000 and others are 0.001, the network becomes unstable. Layer Norm takes all the activations for a single token and rescales them so they have a mean of 0 and a variance of 1. This keeps the numbers in a predictable range, which makes training much faster and smoother.


**6. What is Inductive Bias?**

It is the "built-in" assumption a model has about the data. CNNs assume locality (pixels near each other matter). Transformers have almost zero inductive bias; they assume nothing about the order or structure. This makes them more flexible but means they require massive amounts of data to learn those structures from scratch. Non-LLM methods (like XGBoost or CNNs) are preferred when data is scarce or when the data has a very specific structure (like tabular data) that matches the model's bias.


**7: How does the model know the difference between "The dog bit the man" and "The man bit the dog" if it processes everything in parallel?**

This is strictly the job of Positional Encodings. Without them, the Transformer is a "Bag of Words" model and would see both sentences as identical.

**8: Why do we divide by $\sqrt{d_k}$ in the attention formula?**

This is called Scaling. For large values of $d_k$, the dot product grows very large, pushing the Softmax into regions where the gradient is extremely small. Dividing by $\sqrt{d_k}$ keeps the gradients stable.+1

**9: Is the Encoder or Decoder more important?**

It depends on the task. Encoder-only (BERT) is best for understanding/classifying text. Decoder-only (GPT) is best for generating text. The original Transformer used both for Translation.



## My Understanding

The Transformer is the ultimate "System Design" success story in AI. It trade-offs Inductive Bias (human-coded assumptions about structure) for Scalability. By making every operation a matrix multiplication, it turned a linguistics problem into a high-performance computing (HPC) problem.


### Transformers — A Clean End-to-End Mental Model



### 1. The Objective
A Transformer language model learns:
$$P(x_t \mid x_1, x_2, ..., x_{t-1})$$
It predicts the next token given previous tokens. Training minimizes cross-entropy loss over large text corpora.

### 2. Input Representation
**Assume:**
* Sequence length: $n$
* Model dimension: $d_{model}$
* Vocabulary size: $V$

#### 2.1 Token Embeddings
Each token is mapped via an embedding matrix:
$$E \in \mathbb{R}^{V \times d_{model}}$$
For a sequence of length $n$:
$$E_{seq} \in \mathbb{R}^{n \times d_{model}}$$
Each row is a learned dense vector.

#### 2.2 Positional Encoding


Self-attention is permutation invariant, so we must inject order. We add positional vectors:
$$P \in \mathbb{R}^{n \times d_{model}}$$
Final input:
$$X = E_{seq} + P$$
Now:
$$X \in \mathbb{R}^{n \times d_{model}}$$
Each row encodes both identity and position in a fused representation.

---

### 3. Self-Attention


From $X \in \mathbb{R}^{n \times d}$, we compute:
$$Q = XW_Q$$
$$K = XW_K$$
$$V = XW_V$$
Where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$. So:
$$Q, K, V \in \mathbb{R}^{n \times d_k}$$

#### 3.1 Attention Computation
$$A = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}$$
Apply softmax row-wise:
$$\alpha = \text{softmax}(A)$$
Then:
$$\text{Attention}(X) = \alpha V$$
**Output shape:** $\in \mathbb{R}^{n \times d_k}$

**Interpretation:**
* Each token becomes a weighted combination of all tokens.
* Attention dynamically constructs a fully connected graph over tokens.

---

### 4. Multi-Head Attention


Instead of one attention head, we use $h$ heads. Each head has its own $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}$. Each head computes:
$$\text{head}_h \in \mathbb{R}^{n \times d_k}$$
Concatenate:
$$\text{Concat}(\text{head}_1, ..., \text{head}_h) \in \mathbb{R}^{n \times (h \cdot d_k)}$$
Typically, $h \cdot d_k = d_{model}$. Then project back:
$$W_O \in \mathbb{R}^{d_{model} \times d_{model}}$$
This allows multiple learned similarity geometries simultaneously.

---

### 5. Feedforward Network (FFN)
After attention:
$$(n \times d_{model}) \rightarrow (n \times d_{ff}) \rightarrow (n \times d_{model})$$
Where typically $d_{ff} = 4 \cdot d_{model}$.
* **Attention** mixes tokens.
* **FFN** transforms features independently per token.

### 6. Residual Connections + LayerNorm
Each sublayer:
$$x = \text{LayerNorm}(x + \text{sublayer}(x))$$
**This:**
* Stabilizes gradients.
* Enables deep stacking (dozens of layers).

### 7. Stacking Layers
Repeat attention + FFN for $L$ layers.
* **Lower layers:** Capture local syntactic structure.
* **Higher layers:** Capture semantic abstractions.
Representations evolve layer-by-layer.

---

### 8. Encoder vs Decoder
* **Encoder-only (e.g., BERT):** Full self-attention (no mask). No generation.
* **Decoder-only (e.g., GPT):** Masked self-attention. Autoregressive generation.
* **Encoder-Decoder (original Transformer):**
    * **Encoder:** Full self-attention.
    * **Decoder:** Masked self-attention + Cross-attention (attend to encoder outputs) + FFN.

### 9. Causal Masking (Decoder)


For generation:
$$A_{ij} = -\infty \quad \text{if } j > i$$
Prevents attending to future tokens.

### 10. Output Projection
Final hidden states $H \in \mathbb{R}^{n \times d_{model}}$. Project to vocabulary:
$$\text{logits} = HW_{vocab}$$
$$W_{vocab} \in \mathbb{R}^{d_{model} \times V}$$
Softmax → next-token probabilities.

---

### 11. KV Cache (Inference Optimization)


During autoregressive decoding:
* Previous $K$ and $V$ per layer are cached.
* Only new token’s $Q, K, V$ are computed.
* Attention is computed against cached keys.
Reduces per-token generation cost from $O(n^2) \rightarrow O(n)$.
Memory cost: $O(L \cdot n \cdot d_k)$. The KV cache is a major long-context bottleneck.

### 12. Computational Complexity
Self-attention cost: $O(n^2 d)$. This quadratic term in sequence length is the primary scaling challenge.

---

### 13. Conceptual Interpretation
A Transformer is:
* A learned dynamic graph builder.
* Over tokens in a sequence.
* Using multiple learned similarity metrics.
* Updating node states iteratively.
It does not memorize token-token tables. It learns low-dimensional representations and similarity operators that generalize.

### 14. Why It Works
Transformers succeed because:
* Attention enables direct long-range interaction.
* Multi-head allows multiple relational subspaces.
* Residuals enable deep stacking.
* Massive data allows compression of language statistics.
* Architecture aligns with structured properties of language.
It is a large, differentiable system for modeling sequence distributions via dynamic relational computation.


---

## Follow up Ideas


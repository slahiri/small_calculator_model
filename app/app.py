"""
Calculator LLM - A tiny transformer that solves English math problems.
https://sid.sh/learn/build-your-first-llm
"""

import json
import math

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model Architecture

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        self.scale = math.sqrt(embed_dim)

    def forward(self, x):
        return self.pos_encoding(self.token_embedding(x) * self.scale)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class CalculatorLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, embed_dim, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        mask = torch.tril(torch.ones(x.size(1), x.size(1))).unsqueeze(0).unsqueeze(0).to(x.device)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.output_proj(self.norm(x))


# Tokenizer

class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.id_to_word = {v: k for k, v in vocab.items()}

    def encode(self, text):
        text = text.lower().strip()
        text = text.replace("+", " plus ").replace("-", " minus ").replace("*", " times ").replace("=", " equals ")
        ids = [self.vocab["[START]"]]
        for word in text.split():
            ids.append(self.vocab.get(word, self.vocab["[UNK]"]))
        ids.append(self.vocab["[END]"])
        return ids

    def decode(self, ids):
        special = {"[PAD]", "[START]", "[END]", "[UNK]"}
        return " ".join(self.id_to_word.get(i, "[UNK]") for i in ids if self.id_to_word.get(i) not in special)


# Load model

print("Loading model...")
with open("config.json") as f:
    config = json.load(f)
with open("vocab.json") as f:
    vocab = json.load(f)

model = CalculatorLLM(
    config["vocab_size"], config["embed_dim"], config["num_heads"],
    config["num_layers"], config["ff_dim"], config["max_seq_len"], config.get("dropout", 0.1)
)
model.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True))
model.eval()
tokenizer = Tokenizer(vocab)
print("Ready!")


# Inference

def solve(problem):
    if not problem or not problem.strip():
        return ""

    problem = problem.lower().strip()
    if not problem.endswith("equals"):
        problem += " equals"

    tokens = tokenizer.encode(problem)[:-1]
    input_ids = torch.tensor([tokens])

    with torch.no_grad():
        for _ in range(10):
            logits = model(input_ids)
            next_token = logits[0, -1].argmax().item()
            if next_token == vocab["[END]"]:
                break
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

    result = tokenizer.decode(input_ids[0].tolist())
    return result.split("equals")[-1].strip() if "equals" in result else result


# Gradio UI

with gr.Blocks(title="Calculator LLM") as demo:
    gr.Markdown(
        """
        # Calculator LLM

        A 105K parameter transformer that solves English math problems.
        [[model]](https://github.com/slahiri/small_calculator_model) [[tutorial]](https://sid.sh/learn/build-your-first-llm)

        **Limitations:**
        - Trained on numbers 0-99 only. Inputs or results >99 may produce errors.
        - Test accuracy: ~98% (trained on a small corpus).
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            problem_input = gr.Textbox(
                label="",
                placeholder="Enter your problem",
                lines=1,
                show_label=False,
            )
            run_btn = gr.Button("Run", variant="primary")

        with gr.Column(scale=1):
            answer_output = gr.Textbox(
                label="",
                placeholder="Answer will appear here",
                lines=1,
                show_label=False,
                interactive=False,
            )

    gr.Examples(
        examples=[
            ["two plus three"],
            ["seven times eight"],
            ["ninety minus forty five"],
            ["nine times nine"],
            ["twenty plus thirty"],
            ["eighty one minus forty"],
        ],
        inputs=problem_input,
        outputs=answer_output,
        fn=solve,
        cache_examples=True,
    )

    run_btn.click(fn=solve, inputs=problem_input, outputs=answer_output, api_name="solve")
    problem_input.submit(fn=solve, inputs=problem_input, outputs=answer_output)

if __name__ == "__main__":
    demo.launch()

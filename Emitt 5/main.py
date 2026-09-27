import numpy as np
import pickle
import os


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class Emitt5:
    def __init__(self, vocab_size, d_model=64, max_len=100):
        self.d_model = d_model
        self.vocab_size = vocab_size

        limit = np.sqrt(6.0 / (d_model + vocab_size))
        self.embedding = np.random.uniform(-limit, limit, (vocab_size, d_model))

        self.pos_embedding = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                self.pos_embedding[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                self.pos_embedding[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))

        limit_att = np.sqrt(6.0 / (d_model + d_model))
        self.W_q = np.random.uniform(-limit_att, limit_att, (d_model, d_model))
        self.W_k = np.random.uniform(-limit_att, limit_att, (d_model, d_model))
        self.W_v = np.random.uniform(-limit_att, limit_att, (d_model, d_model))
        self.W_out = np.random.uniform(-limit_att, limit_att, (d_model, vocab_size))

        self.m_q, self.v_q = np.zeros_like(self.W_q), np.zeros_like(self.W_q)
        self.m_k, self.v_k = np.zeros_like(self.W_k), np.zeros_like(self.W_k)
        self.m_v, self.v_v = np.zeros_like(self.W_v), np.zeros_like(self.W_v)
        self.m_out, self.v_out = np.zeros_like(self.W_out), np.zeros_like(self.W_out)
        self.m_emb, self.v_emb = np.zeros_like(self.embedding), np.zeros_like(self.embedding)
        self.t = 0

    def forward(self, input_indices, learning_weights=None, vessel_gates=None):
        self.input_indices = input_indices
        seq_len = len(input_indices)

        if learning_weights is not None and vessel_gates is not None and vessel_gates["pacemaker_active"]:
            active_embedding = self.embedding + (learning_weights["emb"] * 0.05)
            active_q = self.W_q + (learning_weights["q"] * 0.02)
            active_k = self.W_k + (learning_weights["k"] * 0.02)
            active_v = self.W_v + (learning_weights["v"] * 0.02)
            active_out = self.W_out + (learning_weights["out"] * 0.05)
        else:
            active_embedding = self.embedding
            active_q = self.W_q
            active_k = self.W_k
            active_v = self.W_v
            active_out = self.W_out

        self.X = active_embedding[input_indices] + self.pos_embedding[:seq_len]
        self.Q = np.dot(self.X, active_q)
        self.K = np.dot(self.X, active_k)
        self.V = np.dot(self.X, active_v)

        scores = np.dot(self.Q, self.K.T) / np.sqrt(self.d_model)
        mask = np.tril(np.ones((seq_len, seq_len)))
        self.scores = np.where(mask == 1, scores, -1e9)
        self.attention_weights = softmax(self.scores)

        self.context = np.dot(self.attention_weights, self.V)
        self.probs = softmax(np.dot(self.context, active_out))
        return self.probs

    def backward(self, targets, lr=0.01):
        self.t += 1
        d_logits = self.probs.copy()
        for i, target in enumerate(targets):
            d_logits[i, target] -= 1.0

        dW_out = np.dot(self.context.T, d_logits)
        d_context = np.dot(d_logits, self.W_out.T)

        d_V = np.dot(self.attention_weights.T, d_context)
        d_scores = self.attention_weights * (
                    np.dot(d_context, self.V.T) - np.sum(np.dot(d_context, self.V.T) * self.attention_weights, axis=-1,
                                                         keepdims=True))
        d_scores = np.where(np.tril(np.ones((len(self.input_indices), len(self.input_indices)))) == 1, d_scores,
                            0.0) / np.sqrt(self.d_model)

        dW_q = np.dot(self.X.T, np.dot(d_scores, self.K))
        dW_k = np.dot(self.X.T, np.dot(d_scores.T, self.Q))
        dW_v = np.dot(self.X.T, d_V)

        dX = np.dot(np.dot(d_scores, self.K), self.W_q.T) + np.dot(np.dot(d_scores.T, self.Q), self.W_k.T) + np.dot(d_V,
                                                                                                                    self.W_v.T)
        d_emb = np.zeros_like(self.embedding)
        for i, idx in enumerate(self.input_indices):
            d_emb[idx] += dX[i]

        for grad in [dW_q, dW_k, dW_v, dW_out, d_emb]:
            norm = np.linalg.norm(grad)
            if norm > 1.0: grad *= (1.0 / (norm + 1e-6))

        params = [self.W_q, self.W_k, self.W_v, self.W_out, self.embedding]
        ms = [self.m_q, self.m_k, self.m_v, self.m_out, self.m_emb]
        vs = [self.v_q, self.v_k, self.v_v, self.v_out, self.v_emb]
        grads = [dW_q, dW_k, dW_v, dW_out, d_emb]

        for i in range(len(params)):
            ms[i] = 0.9 * ms[i] + 0.1 * grads[i]
            vs[i] = 0.999 * vs[i] + 0.001 * (grads[i] ** 2)
            params[i] -= lr * (ms[i] / (1 - 0.9 ** self.t)) / (np.sqrt(vs[i] / (1 - 0.999 ** self.t)) + 1e-8)

    def generate(self, start_char, char_to_idx, idx_to_char, learning_weights=None, vessel_gates=None):
        curr = [char_to_idx[start_char]]
        out = start_char
        for _ in range(60):
            p = self.forward(curr, learning_weights, vessel_gates)[-1]
            nxt = np.argmax(p)
            out += idx_to_char[nxt]
            if idx_to_char[nxt] == ".": break
            curr.append(nxt)
        return out


if __name__ == "__main__":
    print("--- EMITT 5 ---")

    text = "Benim adim Emitt."
    chars = sorted(list(set(text)))
    c2i = {ch: i for i, ch in enumerate(chars)}
    i2c = {i: ch for i, ch in enumerate(chars)}

    X = [c2i[ch] for ch in text][:-1]
    Y = [c2i[ch] for ch in text][1:]

    core_file = "emitt_core.pkl"
    learning_file = "emitt_learning.pkl"
    vessels_file = "emitt_vessels.pkl"
    pacemaker_file = "emitt_pacemaker.pkl"

    emitt = Emitt5(vocab_size=len(chars))

    if os.path.exists(core_file) and os.path.exists(learning_file) and os.path.exists(vessels_file) and os.path.exists(
            pacemaker_file):
        with open(core_file, "rb") as f:
            core_state = pickle.load(f)
            emitt.embedding = core_state["emb"]
            emitt.W_q = core_state["q"]
            emitt.W_k = core_state["k"]
            emitt.W_v = core_state["v"]
            emitt.W_out = core_state["o"]
            emitt.t = core_state["t"]

        with open(learning_file, "rb") as f:
            learning_weights = pickle.load(f)

        with open(vessels_file, "rb") as f:
            vessel_gates = pickle.load(f)

        with open(pacemaker_file, "rb") as f:
            pacemaker_trigger = pickle.load(f)
            vessel_gates["pacemaker_active"] = pacemaker_trigger["signal"]

        print("[HEARTBEAT] All 4 binary modular structures synchronized successfully.")
    else:
        print("[HEARTBEAT] Building complete 4-tier neural block diagram...")

        for epoch in range(3001):
            emitt.forward(X)
            emitt.backward(Y)

        with open(core_file, "wb") as f:
            pickle.dump({"emb": emitt.embedding, "q": emitt.W_q, "k": emitt.W_k, "v": emitt.W_v, "o": emitt.W_out,
                         "t": emitt.t}, f)
        print(f"[CORTEX] 1. Main brain weights saved to '{core_file}'")

        learning_weights = {
            "emb": np.random.uniform(-0.01, 0.01, emitt.embedding.shape),
            "q": np.random.uniform(-0.01, 0.01, emitt.W_q.shape),
            "k": np.random.uniform(-0.01, 0.01, emitt.W_k.shape),
            "v": np.random.uniform(-0.01, 0.01, emitt.W_v.shape),
            "out": np.random.uniform(-0.01, 0.01, emitt.W_out.shape)
        }
        with open(learning_file, "wb") as f:
            pickle.dump(learning_weights, f)
        print(f"[HIPPOCAMPUS] 2. Dynamic learning data locked into '{learning_file}'")

        vessel_gates = {"vessel_flow_rate": 0.02}
        with open(vessels_file, "wb") as f:
            pickle.dump(vessel_gates, f)
        print(f"[VESSELS] 3. Structural bridge network mapped into '{vessels_file}'")

        pacemaker_trigger = {"signal": True}
        with open(pacemaker_file, "wb") as f:
            pickle.dump(pacemaker_trigger, f)
        print(f"[PACEMAKER] 4. Biological electric valve triggered into '{pacemaker_file}'")
        vessel_gates["pacemaker_active"] = True

    print("\n--- EMITT ---")
    output = emitt.generate("B", c2i, i2c, learning_weights, vessel_gates)
    print(f"Emitt 5: '{output}'")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0

        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys):
        """
        :param queries: A 3d tensor with shape of [N, T_q, C_q]
        :param keys: A 3d tensor with shape of [N, T_k, C_k]

        :return: A 3d tensor with shape of (N, T_q, C)

        """
        Q = self.linear_q(queries)  # (N, T_q, C)
        K = self.linear_k(keys)  # (N, T_k, C)
        V = self.linear_v(keys)  # (N, T_k, C)

        # Split and Concat
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5  # (h*N, T_q, T_k)

        # Key Masking
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_k)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (h*N, T_q, T_k)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # (h*N, T_q, T_k)

        # Causality - Future Blinding
        diag_vals = torch.ones_like(matmul_output[0, :, :])  # (T_q, T_k)
        tril = torch.tril(diag_vals)  # (T_q, T_k)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  # (h*N, T_q, T_k)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings,
                                       matmul_output_m1)  # (h*N, T_q, T_k)

        # Activation
        matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k)

        # Query Masking
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_q)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  # (h*N, T_q, T_k)
        matmul_output_qm = matmul_output_sm * query_mask

        # Dropout
        matmul_output_dropout = self.dropout(matmul_output_qm)

        # Weighted Sum
        output_ws = torch.bmm(matmul_output_dropout, V_)  # ( h*N, T_q, C/h)

        # Restore Shape
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # (N, T_q, C)

        # Residual Connection
        output_res = output + queries

        return output_res

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ---------------- helpers ----------------
def extract_last_by_length(data: torch.Tensor, lengths: torch.LongTensor):
    """ data: (B, T, D), lengths: (B,) """
    idx = (lengths - 1).clamp(min=0)
    batch_idx = torch.arange(data.size(0), device=data.device)
    return data[batch_idx, idx, :]


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        time = time.float().unsqueeze(1)
        out = time * emb.unsqueeze(0)
        return torch.cat([out.sin(), out.cos()], dim=-1)


# ---------------- Cross-Attention Diffuser ----------------
class CrossAttentionDiffuser(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond, t_emb):
        """
        x: (B,D)
        cond: (B,L,D) or (B,D)
        t_emb: (B,D)
        """
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)  # (B,1,D)
        q = (x + t_emb).unsqueeze(1)  # (B,1,D)
        out, _ = self.attn(q, cond, cond, need_weights=False)
        out = out.squeeze(1)
        x = self.norm1(x + self.dropout(out))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x


# ---------------- Main Model ----------------
class Tenc(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, diffuser_type, device,
                 num_heads=4, transformer_layers=2):
        """
        兼容原调用形式:
        Tenc(hidden_size, item_num, state_size, dropout, diffuser_type, device)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.dropout = dropout
        self.diffuser_type = diffuser_type
        self.device = device

        # ---- Embeddings ----
        self.item_embeddings = nn.Embedding(item_num + 1, hidden_size, padding_idx=item_num)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        self.positional_embeddings = nn.Embedding(state_size, hidden_size)
        nn.init.normal_(self.positional_embeddings.weight, 0, 0.02)
        self.none_embedding = nn.Parameter(torch.zeros(1, hidden_size))
        nn.init.xavier_uniform_(self.none_embedding)

        self.emb_dropout = nn.Dropout(dropout)

        # ---- Transformer Encoder ----
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        self.enc_ln = nn.LayerNorm(hidden_size)

        # ---- Step Embedding ----
        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        # ---- Diffuser ----
        if diffuser_type == 'mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size)
            )
        elif diffuser_type == 'mlp2':
            self.diffuser = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size)
            )
        else:
            # new: cross-attention diffuser (default)
            self.diffuser = CrossAttentionDiffuser(hidden_size, num_heads=num_heads, dropout=dropout)

    # ---------- Embedding helper ----------
    def cacu_x(self, x):
        return self.item_embeddings(x)

    # ---------- Encode user history ----------
    def get_h(self, states, len_states):
        B, T = states.shape
        device = states.device
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.item_embeddings(states) + self.positional_embeddings(pos_ids)
        x = self.emb_dropout(x)
        pad_mask = (states == self.item_num)  # True where pad
        out = self.encoder(x, src_key_padding_mask=pad_mask)
        out = self.enc_ln(out)
        h = extract_last_by_length(out, len_states)
        return h  # (B,D)

    # ---------- Apply dropout mask on h ----------
    def cacu_h(self, states, len_states, p):
        h = self.get_h(states, len_states)
        B, D = h.shape
        keep = torch.bernoulli(torch.ones(B, device=h.device) * (1.0 - p)).unsqueeze(1).expand(-1, D)
        none_vec = self.none_embedding.expand(B, -1)
        return h * keep + none_vec * (1.0 - keep)

    # ---------- Conditional forward ----------
    def forward(self, x, h, step):
        t = self.step_mlp(step)
        if isinstance(self.diffuser, CrossAttentionDiffuser):
            # Use h as condition for cross-attention
            out = self.diffuser(x, h, t)
        else:
            out = self.diffuser(torch.cat([x, h, t], dim=1))
        return out

    # ---------- Unconditional forward ----------
    def forward_uncon(self, x, step):
        t = self.step_mlp(step)
        none = self.none_embedding.expand(x.size(0), -1)
        if isinstance(self.diffuser, CrossAttentionDiffuser):
            out = self.diffuser(x, none, t)
        else:
            out = self.diffuser(torch.cat([x, none, t], dim=1))
        return out

    # ---------- Inference ----------
    def predict(self, states, len_states, diff):
        h = self.get_h(states, len_states)
        x = diff.sample(self.forward, self.forward_uncon, h)
        scores = torch.matmul(x, self.item_embeddings.weight[:self.item_num].T)
        return scores
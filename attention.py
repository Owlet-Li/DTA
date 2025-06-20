import torch.nn as nn
import torch
import math
import numpy as np

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
        """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp

        h_flat = h.reshape(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.reshape(-1, input_dim)  # (batch_size*n_query)*input_dim

        shape_k = (batch_size, target_size, -1)
        shape_q = (batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # batch_size*targets_size*key_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))  # batch_size*n_query*targets_size
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            mask = mask.view(batch_size, -1, target_size).expand_as(U)  # copy for n_heads times
            U[mask.bool()] = -1e8
        attention = torch.softmax(U, dim=-1)  # batch_size*n_query*targets_size
        logp_list = torch.log_softmax(U, dim=-1)  # batch_size*n_query*targets_size

        probs = attention

        return probs, logp_list

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
        """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp

        h_flat = h.contiguous().view(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.contiguous().view(-1, input_dim)  # (batch_size*n_query)*input_dim
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(h_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            mask = mask.view(1, batch_size, -1, target_size).expand_as(U)  # copy for n_heads times
            # U[mask.bool()] = -np.inf
            U[mask.bool()] = -np.inf
        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            attnc = attention.clone()
            attnc[mask.bool()] = 0
            attention = attnc
        # print(attention)

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(batch_size, n_query, self.embedding_dim)

        return out  # batch_size*n_query*embedding_dim

class GateFFNDense(nn.Module):
    def __init__(self, model_dim, hidden_unit=512):
        super(GateFFNDense, self).__init__()
        self.W = nn.Linear(model_dim, hidden_unit, bias=False)
        self.V = nn.Linear(model_dim, hidden_unit, bias=False)
        self.W2 = nn.Linear(hidden_unit, model_dim, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, hidden_states):
        hidden_act = self.act(self.W(hidden_states))
        hidden_linear = self.V(hidden_states)
        hidden_states = hidden_act * hidden_linear
        hidden_states = self.W2(hidden_states)
        return hidden_states

class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())

class GateFFNLayer(nn.Module):
    def __init__(self, model_dim):
        super(GateFFNLayer, self).__init__()
        self.DenseReluDense = GateFFNDense(model_dim)
        self.layer_norm = Normalization(model_dim)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        return forwarded_states

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = GateFFNLayer(embedding_dim)

    def forward(self, src, mask=None):
        h0 = src
        h = self.normalization1(src)
        h = self.multiHeadAttention(q=h, mask=mask)
        h = h + h0
        h1 = h
        h = self.feedForward(h)
        h = h + h1
        return h

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(embedding_dim, n_head)
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.feedForward = GateFFNLayer(embedding_dim)
        self.normalization1 = Normalization(embedding_dim)
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, dec_self_attn_mask, dec_enc_attn_mask):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization2(memory)
        h = self.multiHeadAttention(q=tgt, h=memory, mask=dec_enc_attn_mask)
        h = h + h0
        h1 = h
        h = self.feedForward(h)
        h = h + h1
        return h

class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=4, n_layer=2):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src

class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=4, n_layer=2):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, dec_self_attn_mask=None, dec_enc_attn_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, dec_self_attn_mask, dec_enc_attn_mask)
        return tgt

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.sum(dim=2).size()
    batch_size, len_k = seq_k.sum(dim=2).size()
    # eq(zero) is PAD token
    pad_attn_mask_k = seq_q.eq(0).all(2).data.eq(1).unsqueeze(1)  # batch_size x 1 x len_q, one is masking
    pad_attn_mask_q = seq_k.eq(0).all(2).data.eq(1).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    pad_attn_mask_k = pad_attn_mask_k.expand(batch_size, len_k, len_q).permute(0, 2, 1)
    pad_attn_mask_q = pad_attn_mask_q.expand(batch_size, len_q, len_k)
    return ~torch.logical_and(~pad_attn_mask_k, ~pad_attn_mask_q)  # batch_size x len_q x len_k

class AttentionNet(nn.Module):
    def __init__(self, agent_input_dim, task_input_dim, embedding_dim):
        super(AttentionNet, self).__init__()
        self.agent_embedding = nn.Linear(agent_input_dim, embedding_dim)
        self.task_embedding = nn.Linear(task_input_dim, embedding_dim)
        self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)

        self.taskEncoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.crossDecoder1 = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
        self.crossDecoder2 = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
        self.agentEncoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.globalDecoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
        self.pointer = SingleHeadAttention(embedding_dim)

    def encoding_tasks(self, task_inputs, mask=None):
        task_embedding = self.task_embedding(task_inputs)
        task_encoding = self.taskEncoder(task_embedding, mask)
        embedding_dim = task_encoding.size(-1)
        mean_mask = mask[:,0,:].unsqueeze(2).repeat(1, 1, embedding_dim)
        compressed_task = torch.where(mean_mask, torch.nan, task_embedding)
        aggregated_tasks = torch.nanmean(compressed_task, dim=1).unsqueeze(1)
        return aggregated_tasks, task_encoding

    def encoding_agents(self, agents_inputs, mask=None):
        agents_embedding = self.agent_embedding(agents_inputs)
        agents_encoding = self.agentEncoder(agents_embedding, mask)
        embedding_dim = agents_encoding.size(-1)
        mean_mask = mask[:,0,:].unsqueeze(2).repeat(1, 1, embedding_dim)
        compressed_task = torch.where(mean_mask, torch.nan, agents_embedding)
        aggregated_agents = torch.nanmean(compressed_task, dim=1).unsqueeze(1)
        return aggregated_agents, agents_encoding

    def forward(self, tasks, agents, global_mask, index):
        task_mask = get_attn_pad_mask(tasks, tasks)
        agent_mask = get_attn_pad_mask(agents, agents)
        task_agent_mask = get_attn_pad_mask(tasks, agents)
        agent_task_mask = get_attn_pad_mask(agents, tasks)
        aggregated_task, task_encoding = self.encoding_tasks(tasks, mask=task_mask)
        aggregated_agents, agents_encoding = self.encoding_agents(agents, mask=agent_mask)
        task_agent_feature = self.crossDecoder1(task_encoding, agents_encoding, None, task_agent_mask)
        agent_task_feature = self.crossDecoder2(agents_encoding, task_encoding, None, agent_task_mask)
        current_state1 = torch.gather(agent_task_feature, 1, index.repeat(1, 1, agent_task_feature.size(2)))
        current_state = self.fusion(torch.cat((current_state1, aggregated_task, aggregated_agents), dim=-1))
        current_state_prime = self.globalDecoder(current_state, task_agent_feature, None, global_mask)
        probs, logps = self.pointer(current_state_prime, task_agent_feature, mask=global_mask)
        logps = logps.squeeze(1)
        probs = probs.squeeze(1)
        return probs, logps
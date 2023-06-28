

import torch
import torch.nn as nn
import argparse


class BiRNN(nn.Module):
    def __init__(self, opts):
        super(BiRNN, self).__init__()

        self.norm_a = nn.BatchNorm1d(opts.feature_dim_a, affine=False, eps=1e-6)
        self.norm_v = nn.BatchNorm1d(opts.feature_dim_v, affine=False, eps=1e-6)
        self.norm_t = nn.BatchNorm1d(opts.feature_dim_t, affine=False, eps=1e-6)

        self.project_a = nn.Sequential(*[nn.Linear(opts.feature_dim_a, opts.feature_dim_a), nn.ReLU(), nn.Linear(opts.feature_dim_a, opts.hidden_dim_a)])        
        self.project_v = nn.Sequential(*[nn.Linear(opts.feature_dim_v, opts.feature_dim_v), nn.ReLU(), nn.Linear(opts.feature_dim_v, opts.hidden_dim_v)])        
        self.project_t = nn.Sequential(*[nn.Linear(opts.feature_dim_t, opts.feature_dim_t), nn.ReLU(), nn.Linear(opts.feature_dim_t, opts.hidden_dim_t)])

        self.rnn = nn.LSTM(
            input_size    = (opts.hidden_dim_a+opts.hidden_dim_v+opts.hidden_dim_t)*2, 
            hidden_size   = opts.hidden_size,
            num_layers    = opts.num_layers, 
            batch_first   = True,
            bidirectional = True
            )

        self.classifier = nn.Linear(opts.hidden_size * 2, 7)

    def forward(self, x_a, x_v, x_t, id_num):
        N, L, _, = x_a.shape

        x_a = self.norm_a(x_a.view(N * L, -1))
        x_v = self.norm_v(x_v.view(N * L, -1))
        x_t = self.norm_t(x_t.view(N * L, -1))

        x_a = self.project_a(x_a)
        x_v = self.project_v(x_v)
        x_t = self.project_t(x_t)

        x_a = x_a.view(N, L, -1)
        x_v = x_v.view(N, L, -1)
        x_t = x_t.view(N, L, -1)
        
        x_m = torch.cat([x_a, x_v, x_t], dim=2)
        x_m_copy = x_m.clone()

        id_num = id_num.unsqueeze(-1).unsqueeze(0)    
        x_m = x_m * id_num
        x_m_copy = x_m_copy * (1 - id_num)

        x_A = torch.cat((x_m, x_m_copy), dim=2)
        x_B = torch.cat((x_m_copy, x_m), dim=2)

        out_A, (_, _) = self.rnn(x_A)
        out_B, (_, _) = self.rnn(x_B)   
        
        out_pred_A = self.classifier(out_A)
        out_pred_B = self.classifier(out_B)
        out_pred   = out_pred_A * id_num + out_pred_B * (1-id_num)

        out_pred = out_pred.view(N, L, -1)

        return out_pred       


parser = argparse.ArgumentParser(description='MERC2023')  
parser.add_argument('--feature_dim_a', type=int, default=256)
parser.add_argument('--feature_dim_v', type=int, default=256)
parser.add_argument('--feature_dim_t', type=int, default=256)   
parser.add_argument('--hidden_dim_a', type=int, default=32)
parser.add_argument('--hidden_dim_v', type=int, default=32)
parser.add_argument('--hidden_dim_t', type=int, default=32)   
parser.add_argument('--hidden_size',  type=int, default=128)   
parser.add_argument('--num_layers',   type=int, default=2)  
opts = parser.parse_args()

bs = 16
seq_len = 50
f_a = torch.rand(bs, seq_len, opts.feature_dim_a)
f_v = torch.rand(bs, seq_len, opts.feature_dim_v)
f_t = torch.rand(bs, seq_len, opts.feature_dim_t)
id_num = torch.tensor([0 if i%2==0 else 1 for i in range(seq_len)])

net = BiRNN(opts)
out_pred = net(f_a, f_v, f_t, id_num)
print(out_pred.shape)



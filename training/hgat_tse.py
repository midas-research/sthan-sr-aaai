from torch_geometric import nn
import torch
import torch.nn.functional as F
import torch.nn


class Attention(torch.nn.Module):

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = torch.nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = torch.nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.tanh = torch.nn.Tanh()
        self.ae = torch.nn.Parameter(torch.FloatTensor(95,1,1))
        self.ab = torch.nn.Parameter(torch.FloatTensor(95,1,1))

    def forward(self, query, context):
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        mix = attention_weights*(context.permute(0,2,1))
        delta_t = torch.flip(torch.arange(0, query_len), [0]).type(torch.float32).to('cuda')
        delta_t = delta_t.repeat(95,1).reshape(95,1,query_len)
        bt = torch.exp(-1*self.ab * delta_t)
        term_2 = F.relu(self.ae * mix * bt)
        mix = torch.sum(term_2+mix, -1).unsqueeze(1)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

class gru(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(gru, self).__init__()
        self.gru1 = torch.nn.GRU(input_size = input_size, hidden_size=hidden_size, batch_first=True)
    def forward(self, inputs):
        full, last  = self.gru1(inputs)
        # print(full.size())
        # print(last.size())
        return full,last


class HGAT(torch.nn.Module):
    def __init__(self, tickers):
        super(HGAT, self).__init__()
        self.tickers = tickers
        self.grup = gru(5,32) 
        self.attention = Attention(32)
        self.liear2 = torch.nn.Linear(32,32)
        self.hatt1 = nn.HypergraphConv(32, 32, use_attention=True, heads=4, concat=False, negative_slope=0.2, dropout=0.5, bias=True)
        self.hatt2 = nn.HypergraphConv(32, 32, use_attention=True, heads=1, concat=False, negative_slope=0.2, dropout=0.5, bias=True)
        self.liear = torch.nn.Linear(32,1)

    def forward(self,price_input,e):
        context,query  = self.grup(price_input)
        query = query.reshape(95,1,32)
        output, weights = self.attention(query, context)
        output = (output.reshape((95,32)))
        output = self.liear2(output)
        x = (F.leaky_relu(self.hatt1(output,e), 0.2))
        x = (F.leaky_relu(self.hatt2(x,e), 0.2))
        return (self.liear(x))

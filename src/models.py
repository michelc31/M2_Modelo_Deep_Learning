import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Batch First: (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, lengths=None):
        # x: (batch, seq_len, input_dim)
        
        # If we wanted to use pack_padded_sequence:
        # if lengths is not None:
        #     x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM output: output, (h_n, c_n)
        # output: (batch, seq_len, hidden_dim)
        # h_n: (num_layers, batch, hidden_dim)
        out, (h_n, c_n) = self.lstm(x)
        
        # We take the last hidden state of the last layer
        # h_n[-1] is (batch, hidden_dim)
        last_hidden = h_n[-1]
        
        logits = self.fc(last_hidden)
        return logits

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super(CNNLSTM, self).__init__()
        
        # 1D Convolution to extract temporal features
        # Input to Conv1d: (batch, channels, seq_len) -> We need to transpose x
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(0.3)
        
        # LSTM
        # Input to LSTM will be 64 channels
        self.lstm = nn.LSTM(64, hidden_dim, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, lengths=None):
        # x: (batch, seq_len, input_dim)
        
        # Transpose for Conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout_cnn(x)
        
        # Transpose back for LSTM: (batch, seq_len_new, 64)
        x = x.transpose(1, 2)
        
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        
        logits = self.fc(last_hidden)
        return logits


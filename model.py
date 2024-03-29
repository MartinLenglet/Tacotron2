import numpy as np
from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

global hps

def get_mask_from_lengths(lengths):
    max_lengths = max(lengths); nb_lengths=len(lengths)
    ids=np.arange(0, max_lengths)
    mask=(ids<np.reshape(lengths,(nb_lengths,1)))
    mask=torch.from_numpy(mask).cuda()
    return mask

def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)
    
class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention

class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, dim_data * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        # BL : T-in and max_time are the same ???
        RETURNS
        -------
        alignment (batch, max_time)
        """
        # Linear Projection of attention hidden state on attention_dim dimensions
        processed_query = self.query_layer(query.unsqueeze(1))

        # Calculate Location features from last attention weights and cumulative weights
        # (attention_location_n_filters 1D-Conv + linear projection on on attention_dim dimensions)
        processed_attention_weights = self.location_layer(attention_weights_cat)
        
        # processed_attention_weights and processed_memory are the same size (batch, number_characteres, attention_dim)
        # processed_query.size() = (batch, 1, attention_dim), and is summed on all tensor rows
        # result is passed in tanh, en linear projected on 1 dimension
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))
        
        energies = energies.squeeze(-1)
        return energies

    # Automaticaly called when Attention(...) is called in Decoder.__init__
    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        # alignment.size() = (batch, number_characteres, 1): 1 value per encoder_output_state
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        # Softmax function to compute probabilities distribution over encoder_states
        attention_weights = F.softmax(alignment, dim=1)

        # Calculate attention context vector: weighted sum of encoder_states
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hps):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hps.dim_data, hps.postnet_embedding_dim,
                         kernel_size=hps.postnet_kernel_size, stride=1,
                         padding=int((hps.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hps.postnet_embedding_dim))
        )

        for i in range(1, hps.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hps.postnet_embedding_dim,
                             hps.postnet_embedding_dim,
                             kernel_size=hps.postnet_kernel_size, stride=1,
                             padding=int((hps.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hps.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hps.postnet_embedding_dim, hps.dim_data,
                         kernel_size=hps.postnet_kernel_size, stride=1,
                         padding=int((hps.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hps.dim_data))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hps):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hps.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hps.encoder_embedding_dim,
                         hps.encoder_embedding_dim,
                         kernel_size=hps.encoder_kernel_size, stride=1,
                         padding=int((hps.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hps.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.lstm_hidden_dim = int(hps.encoder_embedding_dim / 2)
        self.lstm = nn.LSTM(hps.encoder_embedding_dim,
                            self.lstm_hidden_dim, 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, lstm_state_ini):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x, lstm_state_ini) # hidden/context to be biaised by the style component???
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs

    def inference(self, x, lstm_state_ini):
        for conv in self.convolutions:
            x = F.relu(conv(x))
            #x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x, lstm_state_ini)
        return outputs
    
class Decoder(nn.Module):
    def __init__(self, hps):
        super(Decoder, self).__init__()
        self.dim_data = hps.dim_data
        self.n_frames_per_step = hps.n_frames_per_step
        self.encoder_embedding_dim = hps.encoder_embedding_dim
        self.attention_rnn_dim = hps.attention_rnn_dim
        self.decoder_rnn_dim = hps.decoder_rnn_dim
        self.prenet_dim = hps.prenet_dim
        self.max_decoder_steps = hps.max_decoder_steps
        self.gate_threshold = hps.gate_threshold
        self.p_attention_dropout = hps.p_attention_dropout
        self.p_decoder_dropout = hps.p_decoder_dropout

        # Use last generated mel-spectro in new timestep
        self.prenet = Prenet(
            hps.dim_data * hps.n_frames_per_step,
            [hps.prenet_dim, hps.prenet_dim])
        # 1st Layer LSTM: Attention Layer
        self.attention_rnn = nn.LSTMCell(
            hps.prenet_dim + hps.encoder_embedding_dim,
            hps.attention_rnn_dim)
        # Calculate alignement and attention context vector
        self.attention_layer = Attention(
            hps.attention_rnn_dim, hps.encoder_embedding_dim,
            hps.attention_dim, hps.attention_location_n_filters,
            hps.attention_location_kernel_size)
        # 2nd Layer LSTM: Decoder
        self.decoder_rnn = nn.LSTMCell(
            hps.attention_rnn_dim + hps.encoder_embedding_dim,
            hps.decoder_rnn_dim, 1)
        # Mel-spectro projection
        self.linear_projection = LinearNorm(
            hps.decoder_rnn_dim + hps.encoder_embedding_dim,
            hps.dim_data * hps.n_frames_per_step)
        # Stop token
        self.gate_layer = LinearNorm(
            hps.decoder_rnn_dim + hps.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')


    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(B, self.dim_data * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())
        self.decoder_hidden = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())
        self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(B, self.encoder_embedding_dim).zero_())
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, dim_data, T_out) -> (B, T_out, dim_data)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, dim_data) -> (T_out, B, dim_data)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)

        # if len(gate_outputs)>1:
            # gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
            # gate_outputs = gate_outputs.contiguous()
        gate_outputs = torch.stack(gate_outputs)
        if alignments.shape[0]>1:
            gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        else:
            gate_outputs = gate_outputs.view(1,-1)
            
        # (T_out, B, dim_data) -> (B, T_out, dim_data)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.dim_data)
        # (B, T_out, dim_data) -> (B, dim_data, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output
        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        # First layer input: Concatenate prenet output and attention contexte vector 
        cell_input = torch.cat((decoder_input, self.attention_context), -1)

        # Firt layer LSTM: Attention layer
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))

        # Apply dropout during training
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        # Concatenate cumulative attention weights (to generate location features)
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)

        # Calculate new alignement and attention context vector
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        # Calculate cumulative attention weights (to generate location features for next step)
        self.attention_weights_cum += self.attention_weights

        # Second layer input: concatenate attention hidden states and new attention context
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)

        # Second layer: Decoder Layer
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))

        # Apply dropout during training
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        # Concatenate 2nd layer output with attention context vector
        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)

        # Linear Projections to get mel-spectro and stop token
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        # torch.unsqueeze(input, dim, out=None) → Tensor
        # Returns a new tensor with a dimension of size one inserted at the specified position.
        #The returned tensor shares the same underlying data with this tensor.

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)
        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))
        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        # Initialize decoder output
        decoder_input = self.get_go_frame(memory)
        self.initialize_decoder_states(memory, mask=None)
        mel_outputs, gate_outputs, alignments = [], [], []

        # Generate output while necessary: 1 iteration by output timestep
        while True:
            # Use output from previous timestep (through prenet) for new timestep
            decoder_input = self.prenet(decoder_input)

            # Generate new timestep
            mel_output, gate_output, alignment = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            # Stop Conditions
            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                # Stop token
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                # Maximum decoder steps (ensure this while is not infinite)
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hps):
        super(Tacotron2, self).__init__()
        self.mask_padding = hps.mask_padding
        self.fp16_run = hps.fp16_run
        self.dim_data = hps.dim_data
        self.encoder_embedding_dim = hps.encoder_embedding_dim
        self.prenet_dim = hps.prenet_dim
        self.postnet = hps.postnet
        self.postnet_kernel_size = hps.postnet_kernel_size
        self.postnet_embedding_dim = hps.postnet_embedding_dim
        self.decoder_rnn_dim = hps.decoder_rnn_dim
        self.n_frames_per_step = hps.n_frames_per_step
        self.embedding = nn.Embedding(hps.n_symbols, hps.symbols_embedding_dim)
        std = sqrt(2.0 / (hps.n_symbols + hps.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        self.encoder = Encoder(hps)
        self.decoder = Decoder(hps)
        if hps.postnet:
            self.postnet = Postnet(hps)
        
    def freeze_encoder(self):
        print('Freeze_encoder')
        for name, child in self.encoder.named_children():
 #           print('{} {}'.format(name,child))
            for param in child.parameters():
                param.requires_grad = False

    def parse_batch(self, batch):
        text_in_padded, lg_in, spe_tgt_padded, gate_tgt_padded, lg_tgt, i_nm = batch
        text_in_padded = to_gpu(text_in_padded).long()
        spe_tgt_padded = to_gpu(spe_tgt_padded).float()
        gate_tgt_padded = to_gpu(gate_tgt_padded).float()
        return (
            (text_in_padded, lg_in, spe_tgt_padded, lg_tgt),
            (spe_tgt_padded, gate_tgt_padded), i_nm)

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.dim_data, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies
        return outputs

    def forward(self, inputs):
        #BL: Forward Tacotron with teacher forcing
        text_in, text_lg, spe_tgt, spe_lg = inputs
        [lg_batch,_]=text_in.shape

        # Init LSTM
        h0 = torch.zeros(2, lg_batch, self.encoder.lstm_hidden_dim); h0 = to_gpu(h0).float()
        c0 = torch.zeros(2, lg_batch, self.encoder.lstm_hidden_dim); c0 = to_gpu(c0).float()

        embedded_inputs = self.embedding(text_in).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lg, (h0,c0))

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, spe_tgt, memory_lengths=text_lg)
        if self.postnet:
            mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs) # calcule un residu
        else:
            mel_outputs_postnet = mel_outputs
        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, encoder_outputs],
            spe_lg)

    def inference(self, inputs, seed):
        #BL: Forward Tacotron using previous predictions
        #BL: This seed impacts the dropout and the inference

        # Get inputs args
        text_in = inputs 

        # torch.manual_seed(seed) sets ramdom number generator to a fix value (reproductible random)
        torch.manual_seed(seed)

        # Get input sentence's embedding
        embedded_inputs = self.embedding(text_in).transpose(1, 2)

        # Init LSTM
        h0 = torch.zeros(2, 1, self.encoder.lstm_hidden_dim); h0 = to_gpu(h0).float()
        c0 = torch.zeros(2, 1, self.encoder.lstm_hidden_dim); c0 = to_gpu(c0).float()

        # Encoder
        encoder_outputs = self.encoder.inference(embedded_inputs,(h0,c0))

        # Decoder
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        # Post-treatment if specified
        if self.postnet:	
            mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)
        else:
            mel_outputs_postnet = mel_outputs

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, encoder_outputs])

        return outputs

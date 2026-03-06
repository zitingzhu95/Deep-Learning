"""
S2S Decoder model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN", attention=False):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type
        self.attention = attention

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer based on the "model_type" argument.            #
        #          Supported types (strings): "RNN", "LSTM". Instantiate the        #
        #          appropriate layer for the specified model_type.                  #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #       5) If attention is True, A linear layer to downsize concatenation   #
        #           of context vector and input                                     #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
        
        self.embedding = nn.Embedding(self.output_size ,self.emb_size)

        if self.model_type == "RNN":
            self.rnn = nn.RNN(self.emb_size,self.decoder_hidden_size,batch_first=True)
        elif self.model_type == "LSTM":
            self.rnn = nn.LSTM(self.emb_size,self.decoder_hidden_size,batch_first=True)
        else:
            print("Not the supported model, please use RNN or LSTM")
            
        self.linear1 = nn.Linear(self.decoder_hidden_size,self.output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.dropout = nn.Dropout(dropout)

        if self.attention:
            self.attention_dense = nn.Linear(self.encoder_hidden_size + self.emb_size, self.emb_size)


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def compute_attention(self, hidden, encoder_outputs):
        """ compute attention probabilities given a controller state (hidden) and encoder_outputs using cosine similarity
            as your attention function.

                cosine similarity (q,K) =  q@K.Transpose / |q||K|
                hint |K| has dimensions: N, T
                Where N is batch size, T is sequence length

            Args:
                hidden (tensor): the controller state (dimensions: 1,N, hidden_dim)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention (dimensions: N,T, hidden dim)
            Returns:
                attention: attention probabilities (dimension: N,1,T)
        """

        #############################################################################
        #                              BEGIN YOUR CODE                              #
        # It is recommended that you implement the cosine similarity function from  #
        # the formula given in the docstring. This exercise will build up your     #
        # skills in implementing mathematical formulas working with tensors.        #
        # Alternatively you may use nn.torch.functional.cosine_similarity or        #
        # some other similar function for your implementation.                      #
        #############################################################################
        query = hidden.transpose(0, 1)
        sim = torch.nn.functional.cosine_similarity(query, encoder_outputs, dim=2)
        attention_prob = torch.nn.functional.softmax(sim.unsqueeze(1), dim=-1)

        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return attention_prob

    def forward(self, input, hidden, encoder_outputs=None):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden state of the previous time step from the decoder, dimensions: (1,N,decoder_hidden_size)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention, dimensions: (N,T,encoder_hidden_size)
                attention (Boolean): If True, need to implement attention functionality
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
                hidden (tensor): the state coming out of the hidden unit, dimensions: (1,N,decoder_hidden_size)
                where N is the batch size, T is the sequence length
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       1) Apply the dropout to the embedding layer                         #
        #                                                                           #
        #       2) If attention is true, compute the attention probabilities and    #
        #       use them to do a weighted sum on the encoder_outputs to determine   #
        #       the context vector. The context vector is then concatenated with    #
        #       the output of the dropout layer  as concat(context, dropout_output) #
        #       and is fed into the linear layer you created in the init section.   #
        #       The output of this layer is fed as input vector to your             #
        #       recurrent layer. Refer to the diagram provided in the Jupyter       #
        #       notebook for further clarifications.                                #
        #       note that attention is only applied to the hidden state of LSTM.    #
        #                                                                           #
        #       3) Apply linear layer and log-softmax activation to output tensor   #
        #       before returning it.                                                #
        #                                                                           #
        #       If model_type is LSTM, the hidden variable returns a tuple          #
        #       containing both the hidden state and the cell state of the LSTM.    #
        #############################################################################
        embedded = self.dropout(self.embedding(input))

        if embedded.dim() == 1:
            embedded = embedded.unsqueeze(0) # 如果 batch=1
        if embedded.dim() == 2:
            embedded = embedded.unsqueeze(1) # 正常情况变为 (N, 1, emb)
            
        if self.attention:
            
            h_state = hidden[0] if self.model_type == "LSTM" else hidden
            
            attention_prob = self.compute_attention(h_state,encoder_outputs)
            
            context = torch.bmm(attention_prob, encoder_outputs)
            
            combined = torch.cat((context, embedded), dim=2)
            rnn_input = torch.tanh(self.attention_dense(combined))
            
        else:
            rnn_input = embedded
            
            
        rnn_output, hidden = self.rnn(rnn_input, hidden)

        output = rnn_output.squeeze(1)
        output = self.log_softmax(self.linear1(output))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden

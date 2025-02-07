import torch
import torch.nn as nn
from model_T import TSN
import copy
from torch.nn.init import normal_, constant_
from ops.basic_ops import ConsensusModule
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self,
                 dim,  
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.5,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, 128)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,attn


class Fusion_Network(nn.Module):

    def __init__(self, input_dim, modality, midfusion, dropout, num_segments,norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_segments=num_segments
        self.modality = modality
        self.midfusion = midfusion
        self.dropout = dropout

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        self.selfat = Attention(768)

    def forward(self, inputs):
        outs = []
        if len(self.modality) > 1:  # Fusion
            if self.midfusion == 'concat':
                for m in self.modality:
                    out = inputs[m]
                    out = out.unsqueeze(1)
                    outs.append(out)
                out = torch.cat(outs, dim=1)
                out = self.selfat(out)

                base_out = torch.mean(out[0],1)

            elif self.midfusion == 'context_gating':
                base_out = torch.cat(inputs, dim=1)
                base_out = self.fc1(base_out)
                base_out = self.relu(base_out)
                base_out = self.context_gating(base_out)
            elif self.midfusion == 'multimodal_gating':
                base_out = self.multimodal_gated_unit(inputs)
        else:  # Single modality
            base_out = inputs[0]


        output = {'features': base_out,'attention':out[1]}

        return output
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        return self


class Classification_Network(nn.Module):
    def __init__(self, feature_dim, modality, num_class,
                 dropout, before_softmax, num_segments):
        super().__init__()
        self.num_class = num_class
        self.modality = modality
        self.reshape = True
        self.before_softmax = before_softmax
        self.num_segments = num_segments
        self.dropout = dropout
        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._add_classification_layer(feature_dim)

        if self.dropout > 0:
                    self.dropout_layer = nn.Dropout(p=self.dropout)

    def _add_classification_layer(self, input_dim):

        std = 0.001
        if isinstance(self.num_class, (list, tuple)):  # Multi-task

            self.fc_verb = nn.Linear(input_dim, self.num_class[0])
            self.fc_noun = nn.Linear(input_dim, self.num_class[1])
            normal_(self.fc_verb.weight, 0, std)
            constant_(self.fc_verb.bias, 0)
            normal_(self.fc_noun.weight, 0, std)
            constant_(self.fc_noun.bias, 0)
        else:
            self.fc_action = nn.Linear(input_dim, self.num_class)
            normal_(self.fc_action.weight, 0, std)
            constant_(self.fc_action.bias, 0)
            self.weight = self.fc_action.weight
            self.bias = self.fc_action.bias

    def forward(self, inputs):

        # Snippet-level predictions and temporal aggregation with consensus
        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            # Verb
            base_out_verb = self.fc_verb(inputs)
            if not self.before_softmax:
                base_out_verb = self.softmax(base_out_verb)
            if self.reshape:
                base_out_verb = base_out_verb.view((-1, self.num_segments) + base_out_verb.size()[1:])
            output_verb = self.consensus(base_out_verb)

            # Noun
            base_out_noun = self.fc_noun(inputs)
            if not self.before_softmax:
                base_out_noun = self.softmax(base_out_noun)
            if self.reshape:
                base_out_noun = base_out_noun.view((-1, self.num_segments) + base_out_noun.size()[1:])
            output_noun = self.consensus(base_out_noun)

            output = (output_verb.squeeze(1), output_noun.squeeze(1))

        else:
            base_out = self.dropout_layer(inputs)
            output = self.fc_action(base_out)
        return {'logits': output}


class ARTFNet(nn.Module):
    def __init__(self, num_segments, modality, base_model='ViT',
                 new_length=None, consensus_type='avg', convnet_type='MFDG', before_softmax=True,
                 dropout=0.8, midfusion='concat'):
        super().__init__()

        self.fusion_networks = nn.ModuleList()
        self.fc_list = nn.ModuleList()
        self.fc = None
        self.aux_fc = None
        self.fc_full = None

        self.num_segments = num_segments
        self.modality = modality
        self.base_model = base_model
        self.new_length = new_length
        self.dropout = dropout
        self.before_softmax = before_softmax
        self.consensus_type = consensus_type
        self.convnet_type = convnet_type
        self.midfusion = midfusion

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        self.feature_extract_network = TSN(self.num_segments, self.modality,
                                           self.base_model, self.new_length)

        self.fusion_network = Fusion_Network(768, self.modality, self.midfusion, self.dropout,self.num_segments )

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.feature_extract_network.new_length,
                   consensus_type, self.dropout)))

    @property
    def feature_dim(self):
        if len(self.modality) > 1:
            # return 1024*len(self.modality)
            return 128
        else:
            return 768

    def get_convnet(self, convnet_type):
        name = convnet_type.lower()
        if name == "mfdg":
            model = Fusion_Network(768, self.modality, self.midfusion, self.dropout,self.num_segments)
            return model
        if name == 'fc':
            in_features = self.fc.fc_action.in_features
            incre_classes = self.fc.num_class
            model = Classification_Network(in_features, self.modality, incre_classes,
                                           self.dropout, self.before_softmax, self.num_segments)
            return model
        else:
            raise NotImplementedError("Unknown type {}".format(convnet_type))

    def extract_vector(self, x):
        vit_features = self.feature_extract_network(x)
        modal_feature = self.fusion_network(vit_features)["features"]
        return modal_feature
    

    def forward(self, x, mode='train'):
        vit_features = self.feature_extract_network(x)
        modal_feature = self.fusion_network(vit_features)["features"]
        Attention=self.fusion_network(vit_features)["attention"]

        if mode == 'train':
            out = self.fc(modal_feature)  # {logics: self.fc(features)}
            out.update({"vit_features": vit_features, "fusion_features": modal_feature})
        elif mode == 'test':
            fusion_features, logits = [], []
            fake_logits = []
            for idx, fc in enumerate(self.fc_list):
                ff = self.fusion_networks[idx](vit_features)["features"]
                out = fc(ff)["logits"][:, :4]
                fake_logits.append(fc(ff)["logits"])
                fusion_features.append(ff)
                logits.append(out)
            #fake_logits = torch.cat(fake_logits, 1)
            #print(fake_logits)
            fusion_features = torch.cat(fusion_features, 1)
            logits = torch.cat(logits, 1)
            out = {"logits": logits}
            out.update({"features": vit_features, "fusion_features": fusion_features,'attention':Attention})

        if self.aux_fc is not None:
            aux_logits = self.aux_fc(modal_feature)["logits"]
            out.update({"aux_logits": aux_logits})

        return out
        """
        {
            'features': features
            'logits': logits
        }
        """

    def save_parameter(self):
        self.fusion_networks.append(self.get_convnet(self.convnet_type))
        self.fusion_networks[-1].load_state_dict(self.fusion_network.state_dict())

        if self.fc is not None:
            self.fc_list.append(self.get_convnet('FC'))
            self.fc_list[-1].load_state_dict(self.fc.state_dict())

        # print(self.fc_list.state_dict())

    def _gen_train_fc(self, incre_classes):
        fc = Classification_Network(self.feature_dim, self.modality, incre_classes,
                                    self.dropout, self.before_softmax, self.num_segments)
        del self.fc
        self.fc = fc

        fusion_network=Fusion_Network(768, self.modality, self.midfusion, self.dropout, self.num_segments)
        del self.fusion_network
        self.fusion_network = fusion_network

    def _gen_test_fc(self, total_classes):
        fc_full = Classification_Network(self.feature_dim* len(self.fusion_networks) , self.modality, total_classes,
                                         self.dropout, self.before_softmax, self.num_segments)
        constant_(fc_full.weight, 0)

        if len(self.fc_list) != 0:
            for i, fc in enumerate(self.fc_list):
                nb_output = fc.num_class
                w = self.fc_list[i].state_dict()['weight']
                b = self.fc_list[i].state_dict()['bias']
                weight = copy.deepcopy(w)
                bias = copy.deepcopy(b)
                fc_full.weight.data[nb_output * i: nb_output * (i + 1),
                self.feature_dim * (i): self.feature_dim * (i + 1)] = weight
                fc_full.bias.data[nb_output * i: nb_output * (i + 1)] = bias

        else:
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc_full.weight.data = weight
            fc_full.bias.data = bias

        del self.fc_full
        self.fc_full = fc_full

    def _update_fusion_fc_layer(self, input_dim, output_dim):

        fc1 = nn.Linear(input_dim, output_dim)

        std = 0.001
        normal_(fc1.weight, 0, std)
        constant_(fc1.bias, 0)

        return fc1

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
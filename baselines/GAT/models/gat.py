import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN
from collections import defaultdict

class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False, 
            export_dict=defaultdict(dict)):
        # Input attention layer
        attns = []
        for head_i in range(n_heads[0]):
            scope_name = f"attn_L1_H{head_i}"
            attns.append(layers.attn_head(scope_name, inputs, bias_mat=bias_mat,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False, 
                export_dict=export_dict[scope_name]))
        h_1 = tf.concat(attns, axis=-1)
        
        # Middle attention layers
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for head_i in range(n_heads[i]):
                scope_name = f"attn_L{i + 1}_H{head_i}"
                attns.append(layers.attn_head(scope_name, h_1, bias_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual, 
                    export_dict=export_dict[scope_name]))
            h_1 = tf.concat(attns, axis=-1)
        
        # Last attention layers
        out = []
        for i in range(n_heads[-1]):
            scope_name = f"attn_L{len(n_heads)}_H{i}"
            out.append(layers.attn_head(scope_name, h_1, bias_mat=bias_mat,
                out_sz=nb_classes, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False, 
                export_dict=export_dict[scope_name]))
        logits = tf.add_n(out) / n_heads[-1]
    
        return logits

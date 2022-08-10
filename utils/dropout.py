from utils.modeling import rsetattr
import torch, math

def add_dropout(model: torch.nn.Module,
                dropout_start_perc: float = 0.0,
                dropout_stop_perc: float = 1.0,
                dropout_prob: float = 0.1):

    # Add dropout layers after relu
    dropout_cls = torch.nn.Dropout
    dropout_prev_modules = (torch.nn.ReLU6, torch.nn.ReLU)
    max_pos = len([m for m in model.modules() if isinstance(m, dropout_prev_modules)])
    start_pos = math.floor(dropout_start_perc * max_pos)
    stop_pos = math.floor(dropout_stop_perc * max_pos)
    pos_ind = 0
    for m_name, m in model.named_modules():
        if isinstance(m, dropout_prev_modules):
            pos_ind += 1
            if pos_ind >= start_pos and pos_ind <= stop_pos:
                rsetattr(model, m_name, torch.nn.Sequential(m, dropout_cls(p=dropout_prob)))

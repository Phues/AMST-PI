# This file is based on the original MLA code,
# link: https://github.com/CXianRen/MLA/blob/main/models/fusion_modules.py
# We support fusion methods:
# 1. lsum: Late Sum, from the original MLA, OGM-GE code, this is fusion
# is a bit wired, but for consistency, we keep it.
# 2. msum: MLA Sum, it is the shared head fusion. It is only used in MLA method 
# and MLA related experiments.
# 3. concat: Early Concat, the standard definition of early fusion

import torch
import torch.nn as nn
import torch.nn.functional as F

Fusion_List = ['concat', 'lsum', 'msum']

def gen_fusion_v2(args, input_dim, output_dim, name_list):
    fusion_methods = {
        'lsum': LateSum,
        'msum': MLASum,
        'concat': EarlyConcat
    }
    
    fusion_method = args.fusion_method
    if fusion_method in fusion_methods:
        return fusion_methods[fusion_method](input_dim, output_dim, name_list)
    else:
        raise NotImplementedError("Fusion method not implemented: {}".format(fusion_method))


# ---------------------------------------------------------------------------
# Gradient modification matrix (MLA, eq. 5)
#
# At each alternating step t the shared head's gradient is projected onto the
# subspace orthogonal to the previous modality's mean encoder output.  This
# prevents the head from over-writing cross-modal information it already
# captured ("modality forgetting").
#
# Usage
# -----
#   modifier = GradModifier(encoder_output_dim)
#   # --- end of forward for modality m_prev ---
#   modifier.update(h_mean)          # record m_prev's mean embedding
#   # --- backward for modality m_curr ---
#   modifier.apply_hook(shared_head) # register a one-shot backward hook
#
# The hook fires during loss.backward(), modifies ∇φ L in-place, then
# removes itself so it doesn't affect the joint module or later steps.
# ---------------------------------------------------------------------------
class GradModifier:
    """
    Recursive-Least-Squares gradient projection matrix from MLA (eq. 5).

    Maintains P ∈ R^{s×s} (initialised to I_s).  After seeing the mean
    encoder output h̄ of the *previous* modality, P is updated so that
    multiplying any gradient by P projects it away from the direction of h̄,
    keeping cross-modal information intact.
    """

    def __init__(self, embed_dim: int, alpha: float = 1.0):
        """
        Args:
            embed_dim: dimension s of the encoder output (= shared-head input).
            alpha:     RLS regularisation constant (prevents division by zero).
        """
        self.embed_dim = embed_dim
        self.alpha = alpha
        # P starts as the identity matrix; stored on CPU and moved lazily.
        self.P: torch.Tensor = torch.eye(embed_dim)
        self._hook_handle = None

    # ------------------------------------------------------------------
    # Called at the END of the forward pass for the PREVIOUS modality,
    # before we switch to the next one.
    # ------------------------------------------------------------------
    def update(self, h_mean: torch.Tensor):
        """
        Update the projection matrix given the mean encoder output of the
        modality that was just processed.

        Args:
            h_mean: shape (embed_dim,) — mean of the encoder outputs in the
                    current batch.  Detached from the graph.
        """
        h = h_mean.detach().to(self.P.device).float()   # (s,)
        # q_t  =  P_{t-1} h̄  /  (α + h̄ᵀ P_{t-1} h̄)
        Ph = self.P @ h                                  # (s,)
        denom = self.alpha + h @ Ph                      # scalar
        q = Ph / denom                                   # (s,)
        # P_t  =  P_{t-1}  −  q_t h̄ᵀ P_{t-1}
        self.P = self.P - torch.outer(q, h) @ self.P

    # ------------------------------------------------------------------
    # Called just before loss.backward() for the CURRENT modality.
    # Registers a one-shot hook on the shared head's weight gradient.
    # ------------------------------------------------------------------
    def apply_hook(self, shared_head: nn.Linear):
        """
        Register a backward hook on `shared_head` that left-multiplies the
        weight gradient by P, then immediately removes itself.

        Only the shared head (nn.Linear) is modified; encoder gradients are
        unaffected.
        """
        P = self.P  # captured by closure

        def _hook(grad: torch.Tensor) -> torch.Tensor:
            # grad shape: (output_dim, embed_dim)  — same as weight
            # P shape:    (embed_dim, embed_dim)
            # projected:  (output_dim, embed_dim)
            device = grad.device
            P_dev = P.to(device)
            return grad @ P_dev.t()

        # Register on the weight parameter (bias is left untouched)
        self._hook_handle = shared_head.weight.register_hook(_hook)

    def remove_hook(self):
        """Explicitly remove the hook if it hasn't fired yet."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def reset(self):
        """Reset P to identity (call between epochs if desired)."""
        self.P = torch.eye(self.embed_dim)


# New fusion api

class MLASum(nn.Module):
    # MLA
    """
        the forward is using equal weights for all modalities by default.
        if we want different weights, we need to call get_out_m function
        for each modality, and then sum the output with the weights.

        When grad_modifier is provided (a GradModifier instance), the shared
        FC head's gradients are orthogonalised w.r.t. the previously trained
        modality before each backward pass.  This matches the MLA gradient
        modification mechanism and should only be enabled for the alternating
        module — NOT for the joint module.

        Structure:
        out = w1*FC(e1) + w2*FC(e2) + w3*FC(e3)
    """
    
    def __init__(self, input_dim, output_dim, modality_name_list,
                 grad_modifier: GradModifier = None):
        super(MLASum, self).__init__()
        # W[e1]+ b + W[e2] +b + W[e3] + b = W[e1 + e2 + e3] + c
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.n_modalities = len(modality_name_list)
        self.out_dict = dict()
        # Optional gradient modification (MLA eq. 3-5).
        # Set to a GradModifier instance to enable; None to disable.
        self.grad_modifier: GradModifier | None = grad_modifier
        # Track which modality was active in the most recent forward so the
        # trainer can call modifier.update() with the right embedding.
        self._last_active_modality: str | None = None
        self._last_active_embedding: torch.Tensor | None = None

    def forward(self, embeddings_dict: dict):
        # This might look a bit complicated, but just 
        # for compatibility with our code. 
        # it can be regared as the fusion with same fixed 
        # weights for all modalities.
        # And for the other weights, is computed separately.
        # by using get_out_m function, which returns the
        # the output of the specific modality.
        
        self.out_dict = dict()
        self._last_active_modality = None
        self._last_active_embedding = None

        # embeddings_dict: {modality: tensor}
        # sum the embeddings
        for k, v in embeddings_dict.items():
            if v is None:
                # This part should only be used when 
                # using alternating training.
                self.out_dict[k] = None
            else:
                self.out_dict[k] = self.fc(v)
                # Record the single active modality for gradient modification.
                # In alternating mode exactly one modality is non-None per
                # forward call, so this captures it unambiguously.
                self._last_active_modality = k
                self._last_active_embedding = v.detach().mean(dim=0)
        
        # sum the output
        valid_values = [v for v in self.out_dict.values() if v is not None]
        if not valid_values:
            raise ValueError("All values in out_dict are None.")

        # compute the mean of the valid values
        # this equals to :  out = (out1 + out2 + out3) / n
        out = torch.mean(torch.stack(valid_values), dim=0)
        
        return out
    
    def get_out_m(self, modality_name):
        if modality_name in self.out_dict:
            return self.out_dict[modality_name]
        else:
            raise ValueError("Modality name not found in the output dict")

    # ------------------------------------------------------------------
    # Gradient modification helpers — called by the trainer around each
    # alternating backward pass.
    # ------------------------------------------------------------------
    def prepare_grad_mod(self):
        """
        Register the backward hook for the current step.
        Call this AFTER forward() but BEFORE loss.backward().
        Only has an effect when grad_modifier is set.
        """
        if self.grad_modifier is not None:
            self.grad_modifier.apply_hook(self.fc)

    def finish_grad_mod(self):
        """
        Update P with the mean embedding of the modality that just trained,
        so the next step's projection reflects the most recently seen
        feature subspace.  Call this AFTER the optimizer step.
        Only has an effect when grad_modifier is set.
        """
        if self.grad_modifier is not None and \
                self._last_active_embedding is not None:
            self.grad_modifier.update(self._last_active_embedding)
 
class EarlyConcat(nn.Module):
    """
        This is the standard definition of concatenation.
        Structure:
        out = W[e1, e2, e3] + b = W1[e1] + b/n + W2[e2]+ b/n + W3[e3] + b/n
        where n is the number of modalities.
    """
    def __init__(self, intput_dim, output_dim, modality_name_list):
        super(EarlyConcat, self).__init__()
        self.n_modalities = len(modality_name_list)
        #  W[e1, e2, e3] + b = W1[e1] + b/n + W2[e2]+ b/n + W3[e3] + b/n
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        self.out_layers = nn.ModuleDict()
        for m in modality_name_list:
            self.out_layers[m] = nn.Linear(intput_dim, output_dim, bias=False)
        
        self.out_dict = dict()
        self.modality_name_list = modality_name_list
        self.modality_name_list.sort()
        
    def forward(self, embeddings_dict: dict):
        self.out_dict = dict()
        # embeddings_dict: {modality: tensor}
        # concatenate the embeddings
        #  W[e1, e2, e3] + b = W1[e1] + b/n + W2[e2]+ b/n + W3[e3] + b/n
        for k, v in embeddings_dict.items():
            if v is None:
                # This part should only be used when
                # using alternating training.
                continue
                
            self.out_dict[k] = self.out_layers[k](v) + self.bias/self.n_modalities
    
        # compute the sum of the valid values
        # this equals to :  out = (out1 + out2 + out3)
        out = torch.sum(torch.stack(list(self.out_dict.values())), dim=0)
        return out
    
    def get_out_m(self, modality_name):
        if modality_name in self.out_dict:
            return self.out_dict[modality_name]
        else:
            raise ValueError("Modality name not found in the output dict")

class LateSum(nn.Module):
    """
    Late Sum fusion module.
    Structure:
    out = W[e1] + b1 + W[e2] + b2 + W[e3] + b3
    where n is the number of modalities.
    
    This is what MLA, OGM-GE code is using.
    This is a bit wired, but for consistency, we keep it.
    """
    # late fusion
    # logit = W1[e1] + b1 + W2[e2] + b2 + W3[e3] + b3
    def __init__(self, input_dim, output_dim, modality_name_list: list):
        super(LateSum, self).__init__()
        self.n_modalities = len(modality_name_list)
        self.out_layers = nn.ModuleDict()
        self.out_dict = dict()
        
        for m in modality_name_list:
            self.out_layers[m] = nn.Linear(input_dim, output_dim)
    
    def forward(self, embeddings_dict: dict):
        self.out_dict = dict()
        # embeddings_dict: {modality: tensor}
        for k, v in embeddings_dict.items():
            if v is None:
                # This part should only be used when
                # using alternating training.
                continue
            self.out_dict[k] = self.out_layers[k](v)
        
        out = torch.sum(torch.stack(list(self.out_dict.values())), dim=0)
        return out

    def get_out_m(self, modality_name):
        if modality_name in self.out_dict:
            return self.out_dict[modality_name]
        else:
            raise ValueError("Modality name not found in the output dict")
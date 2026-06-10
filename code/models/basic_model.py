import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18
from dataset.dataset import get_num_classes
from transformers import RobertaModel
from .fusion_modules import gen_fusion_v2, GradModifier, MLASum

from dataset.dataset import DATASET_HAS_VISUAL_LIST, \
    DATASET_HAS_AUDIO_LIST, DATASET_HAS_TEXT_LIST

# modality name
M_TEXT_NAME   = 't'
M_AUDIO_NAME  = 'a'
M_VISUAL_NAME = 'v'

KEY_HELPERS  = 'helpers'
KEY_ENCODERS = 'encoders'
KEY_FUSION   = 'fusion'
KEY_TEXT_TOKENS       = 'tokens'
KEY_TEXT_PADDING_MASK = 'padding_mask'

# ------------------------------------------------------------------
# TCGA modality constants (re-use 'a'/'v' slots)
# ------------------------------------------------------------------
# CLAM patch bag → visual slot ('v')
TCGA_PATCH_DIM  = 1024   # ResNet-50 CLAM feature dimension
TCGA_EMBED_DIM  = 512    # projected embedding fed to the shared head

# Multi-omics vector → audio slot ('a')
TCGA_OMICS_DIM  = 1024   # raw omics feature vector length


# visual encoder, based on ResNet18
class _ResNet18_V(nn.Module):
    def __init__(self, output_dim=512):
        super(_ResNet18_V, self).__init__()
        self.basic_resnet18 = resnet18()
        # define the first conv layer for the model
        self.basic_resnet18.conv1 = nn.Conv2d( 3, # input channel 
                                            64, # output channel
                                              kernel_size=7, stride=2, padding=3, bias=False)
        if output_dim != 512:
            # to compatible with text model output dim (768)
            self.out_conv = nn.Conv2d(512 * self.basic_resnet18.block.expansion,
                                      output_dim, kernel_size=1, stride=1, bias=False)
        
    def forward(self, x):
        # first, we reshape the input tensor (B, C, T, H, W) -> (B*T, C, H, W)
        # which means we stack the frames of the video
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)
        
        # backbone network
        x = self.basic_resnet18(x)

        # compatible with text model output dim
        if hasattr(self, 'out_conv'):
            x = self.out_conv(x)
        
        # recover the original shape
        (_, C, H, W) = x.size()
        x = x.view(B, -1, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # average pooling over the frames
        out = F.adaptive_avg_pool3d(x, 1)
        
        # flatten the tensor
        out = torch.flatten(out, 1)
        return out

# visual encoder, based on ResNet18
class _ResNet18_A(nn.Module):
    def __init__(self, output_dim=512):
        super(_ResNet18_A, self).__init__()
        self.basic_resnet18 = resnet18()
        # define the first conv layer for the model
        self.basic_resnet18.conv1 = nn.Conv2d( 1, # input channel 
                                              64, # output channel
                                              kernel_size=7, stride=2, padding=3, bias=False)
        
        if output_dim != 512:
            # to compatible with text model output dim
            self.out_conv = nn.Conv2d(512 * self.basic_resnet18.block.expansion,
                                      output_dim, kernel_size=1, stride=1, bias=False)
        
    def forward(self, x):
        out = self.basic_resnet18(x)
        
        if hasattr(self, 'out_conv'):
            out = self.out_conv(out)
            
        out = F.adaptive_avg_pool2d(out, 1)

        out = torch.flatten(out, 1)
        return out

# text encoder, based on RoBERTa
class TextEncoder(nn.Module):
    """
    Encodes pre-tokenized text using a pretrained RoBERTa model from Hugging Face.
    The forward method expects input_ids and attention_mask tensors.
    """
    def __init__(self,
                 model_name="roberta-base",
                 fine_tune=False,
                 unfreeze_last_n_layers=2):
        super(TextEncoder, self).__init__()
        # Load the pretrained RoBERTa model.
        self.model = RobertaModel.from_pretrained(model_name)
        
        # Freeze parameters by default.
        for param in self.model.parameters():
            param.requires_grad = False

        # Optionally unfreeze the last n layers.
        total_layers = len(self.model.encoder.layer)
        print("ROBERTA TOTAL LAYERS: ", total_layers)
        if fine_tune:
            for layer_idx in range(total_layers - unfreeze_last_n_layers, total_layers):
                for param in self.model.encoder.layer[layer_idx].parameters():
                    param.requires_grad = True

    def forward(self, input_ids, attention_mask): 
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] embedding (first token).
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb


# ------------------------------------------------------------------
# TCGA encoder 1: CLAM patch-bag encoder (visual slot)
#
# Input : FloatTensor (B, N_patches, PATCH_DIM)
# Output: FloatTensor (B, embed_dim)
#
# Uses a one-layer attention-MIL pooling:
#   scores = softmax( tanh(W1 h) ⊙ sigmoid(W2 h) )
#   z      = Σ scores_i · h_i
#   out    = FC(z)
# This is the standard ABMIL formulation (Ilse et al., 2018) used by CLAM.
# ------------------------------------------------------------------
class CLAMPatchEncoder(nn.Module):
    def __init__(self, patch_dim: int = 1024, embed_dim: int = 512,
                 attention_dim: int = 256):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, embed_dim)

        # Gated attention (ABMIL)
        self.attn_V = nn.Linear(embed_dim, attention_dim)   # tanh branch
        self.attn_U = nn.Linear(embed_dim, attention_dim)   # sigmoid branch
        self.attn_w = nn.Linear(attention_dim, 1)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm     = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, PATCH_DIM)
        Returns:
            (B, embed_dim)
        """
        # Project patches to embed_dim
        h = torch.relu(self.patch_proj(x))               # (B, N, E)

        # Gated attention scores
        A = torch.tanh(self.attn_V(h)) \
            * torch.sigmoid(self.attn_U(h))              # (B, N, A)
        A = self.attn_w(A)                               # (B, N, 1)
        A = torch.softmax(A, dim=1)                      # (B, N, 1)

        # Weighted sum of patch embeddings
        z = (A * h).sum(dim=1)                           # (B, E)
        z = self.norm(self.out_proj(z))
        return z


# ------------------------------------------------------------------
# TCGA encoder 2: Multi-omics encoder (audio slot)
#
# Input : FloatTensor (B, OMICS_DIM)
# Output: FloatTensor (B, embed_dim)
#
# Three-layer MLP with BatchNorm, matching the embedding dimension of
# CLAMPatchEncoder so both modalities reach the shared head with the
# same feature dimension.
# ------------------------------------------------------------------
class OmicsEncoder(nn.Module):
    def __init__(self, omics_dim: int = 1024, embed_dim: int = 512,
                 hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(omics_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, OMICS_DIM)
        Returns:
            (B, embed_dim)
        """
        return self.net(x)
    
# For testing single modality
# Single modality (visual or audio) model, based on ResNet18
class UniR18(nn.Module):
    def __init__(self, args, feature_dim=512):
        super(UniR18, self).__init__()
        
        self.modality = args.modality

        n_classes = get_num_classes(args.dataset)

        if args.modality == 'audio':
            self.net = _ResNet18_A(feature_dim)

        elif args.modality == 'visual':
            self.net = _ResNet18_V(feature_dim)
        else:
            raise NotImplementedError(
                'Incorrect modality: {}!'.format(args.modality))

        self.fc = nn.Linear(feature_dim, n_classes)
        self.feature = None
        
    def forward(self, x):
        self.feature = self.net(x)
        m = self.fc(self.feature)
        return m

    def get_feature_embedding(self):
        return self.feature

# Single modality (text) model, based on RoBERT
class UniBERT(nn.Module):
    def __init__(self, args):
        super(UniBERT, self).__init__()
        
        self.modality = args.modality

        n_classes = get_num_classes(args.dataset)

        self.text_encoder = TextEncoder(model_name="roberta-base", fine_tune=True, unfreeze_last_n_layers=5)
        
        print("BERT model loaded")
        print("n_classes: ", n_classes)
        print("hidden size: ", self.text_encoder.model.config.hidden_size)
        
        self.fc = nn.Linear(self.text_encoder.model.config.hidden_size, n_classes)
        self.feature = None
        
    def forward(self, input_ids, attention_mask):
        m = self.text_encoder(input_ids, attention_mask)
        self.feature = m
        m = self.fc(m)
        return m

    def get_feature_embedding(self):
        # size should be (batch_size, 768) for BERT
        return self.feature


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        

def gen_model(args):
    embedding_dim = 512
    dataset = args.dataset
    class_num = get_num_classes(args.dataset)

    model_dict = nn.ModuleDict({
        KEY_HELPERS:  nn.ModuleDict(),
        KEY_ENCODERS: nn.ModuleDict(),
        KEY_FUSION:   None
    })

    # ------------------------------------------------------------------
    # TCGA: swap standard encoders for domain-specific ones.
    # Visual slot ('v') → CLAMPatchEncoder
    # Audio  slot ('a') → OmicsEncoder
    # ------------------------------------------------------------------
    if dataset == 'TCGA':
        embedding_dim = TCGA_EMBED_DIM

        print("CLAM patch encoder initialized (visual slot)")
        model_dict[KEY_ENCODERS][M_VISUAL_NAME] = CLAMPatchEncoder(
            patch_dim=TCGA_PATCH_DIM, embed_dim=embedding_dim)
        model_dict[KEY_ENCODERS][M_VISUAL_NAME].apply(weight_init)
        model_dict[KEY_HELPERS][M_VISUAL_NAME] = nn.Linear(embedding_dim, class_num)
        model_dict[KEY_HELPERS][M_VISUAL_NAME].apply(weight_init)

        print("Omics MLP encoder initialized (audio slot)")
        model_dict[KEY_ENCODERS][M_AUDIO_NAME] = OmicsEncoder(
            omics_dim=TCGA_OMICS_DIM, embed_dim=embedding_dim)
        model_dict[KEY_ENCODERS][M_AUDIO_NAME].apply(weight_init)
        model_dict[KEY_HELPERS][M_AUDIO_NAME] = nn.Linear(embedding_dim, class_num)
        model_dict[KEY_HELPERS][M_AUDIO_NAME].apply(weight_init)

    else:
        if dataset in DATASET_HAS_TEXT_LIST:
            print("Text model initialized")
            embedding_dim = 768
            model_dict[KEY_ENCODERS][M_TEXT_NAME] = TextEncoder(
                model_name="roberta-base",
                fine_tune=True,
                unfreeze_last_n_layers=5)
            model_dict[KEY_HELPERS][M_TEXT_NAME] = nn.Linear(embedding_dim, class_num)
            model_dict[KEY_HELPERS][M_TEXT_NAME].apply(weight_init)

        if dataset in DATASET_HAS_VISUAL_LIST:
            print("Visual model initialized")
            model_dict[KEY_ENCODERS][M_VISUAL_NAME] = _ResNet18_V(output_dim=embedding_dim)
            model_dict[KEY_ENCODERS][M_VISUAL_NAME].apply(weight_init)
            model_dict[KEY_HELPERS][M_VISUAL_NAME] = nn.Linear(embedding_dim, class_num)
            model_dict[KEY_HELPERS][M_VISUAL_NAME].apply(weight_init)

        if dataset in DATASET_HAS_AUDIO_LIST:
            print("Audio model initialized")
            model_dict[KEY_ENCODERS][M_AUDIO_NAME] = _ResNet18_A(output_dim=embedding_dim)
            model_dict[KEY_ENCODERS][M_AUDIO_NAME].apply(weight_init)
            model_dict[KEY_HELPERS][M_AUDIO_NAME] = nn.Linear(embedding_dim, class_num)
            model_dict[KEY_HELPERS][M_AUDIO_NAME].apply(weight_init)

    modality_name_list = list(model_dict[KEY_ENCODERS].keys())
    modality_name_list.sort()

    model_dict[KEY_FUSION] = gen_fusion_v2(
        args, embedding_dim, class_num, modality_name_list)
    model_dict[KEY_FUSION].apply(weight_init)

    return model_dict


def gen_alt_fusion_with_grad_mod(args, embedding_dim: int) -> MLASum:
    """
    Build the alternating module's shared head (MLASum) with a GradModifier
    attached.  Called by AMST_F_Trainer so the alt fusion head benefits from
    MLA-style gradient orthogonalisation while the joint head does not.

    Args:
        args:          training args (used for dataset / class count).
        embedding_dim: dimension of each encoder's output.

    Returns:
        An MLASum instance whose .grad_modifier is a freshly initialised
        GradModifier of the correct size.
    """
    class_num = get_num_classes(args.dataset)
    modality_name_list = []
    from dataset.dataset import DATASET_HAS_AUDIO_LIST, \
        DATASET_HAS_VISUAL_LIST, DATASET_HAS_TEXT_LIST
    if args.dataset in DATASET_HAS_TEXT_LIST:
        modality_name_list.append(M_TEXT_NAME)
    if args.dataset in DATASET_HAS_VISUAL_LIST:
        modality_name_list.append(M_VISUAL_NAME)
    if args.dataset in DATASET_HAS_AUDIO_LIST:
        modality_name_list.append(M_AUDIO_NAME)
    modality_name_list.sort()

    modifier = GradModifier(embed_dim=embedding_dim)
    head = MLASum(embedding_dim, class_num, modality_name_list,
                  grad_modifier=modifier)
    head.apply(weight_init)
    return head


def forward_encoders(model_dict: nn.ModuleDict, input_dict: dict, use_ws: bool = False):
    if use_ws:
        return forward_encoders_ws(model_dict, input_dict)
    else:
        return forward_encoders_wos(model_dict, input_dict)

stream_dict = {}

def forward_encoders_ws(model_dict: nn.ModuleDict, input_dict: dict):
    """
    This version is forwarding with multiple streams, Which is faster 
    and for multi-gpu speed up. For the easily understanding, refer to:
    'forward_encoders_wos' function.
    """
    # forward each modality
    feature_dict = {}
    global stream_dict
    
    used_devices = set()
    # run each encoder in a separate stream to speed up
    for modality, encoder in model_dict.items():
        device = next(encoder.parameters()).device
        used_devices.add(device)
        
        if modality not in stream_dict:
            stream_dict[modality] = torch.cuda.Stream(device=device)

        if modality == M_TEXT_NAME:
            with torch.cuda.stream(stream_dict[modality]):
                feature_dict[modality] = encoder(input_dict[M_TEXT_NAME][KEY_TEXT_TOKENS], 
                                                input_dict[M_TEXT_NAME][KEY_TEXT_PADDING_MASK])
        else:
            with torch.cuda.stream(stream_dict[modality]):
                feature_dict[modality] = encoder(input_dict[modality])
    for device in used_devices:
        torch.cuda.synchronize(device=device)
    return feature_dict


def forward_encoders_wos(model_dict: nn.ModuleDict, input_dict: dict):
    """
    To forward each modality encoder and get the features(embedding)
    Args:
        model_dict (nn.ModuleDict): Dictionary of encoder modules.
        input_dict (dict): Dictionary of inputs for each modality.
    Returns:
        feature_dict (dict): Dictionary of features for each modality.
    """
    feature_dict = {}   
    # run each encoder in a separate stream to speed up
    for modality, encoder in model_dict.items():        
        if modality == M_TEXT_NAME:
            feature_dict[modality] = encoder(input_dict[M_TEXT_NAME][KEY_TEXT_TOKENS], 
                                            input_dict[M_TEXT_NAME][KEY_TEXT_PADDING_MASK])
        else:
            feature_dict[modality] = encoder(input_dict[modality])
            
    return feature_dict

def forward_fusion(fuser: nn.Module, feature_dict: dict):
    """
    Obviously, fusion forwarding
    """
    return fuser(feature_dict)

def forward_helper(helper_dict: nn.ModuleDict, feature_dict: dict):
    """
    Forward each helper (classifier) for each modality.
    Args:
        helper_dict (nn.ModuleDict): Dictionary of helper modules.
        feature_dict (dict): Dictionary of features for each modality.
        the feature will be detached to avoid gradient flow
    Returns:
        output_dict (dict): Dictionary of outputs for each modality.
    """
    # forward each helper
    output_dict = {}
    for modality, helper in helper_dict.items():
        output_dict[modality] = helper(feature_dict[modality].detach())

    return output_dict

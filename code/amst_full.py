import os

import torch
import torch.nn as nn
import torch.optim as optim

from models.basic_model import (
    M_TEXT_NAME, M_AUDIO_NAME, M_VISUAL_NAME,
    KEY_HELPERS, KEY_ENCODERS, KEY_FUSION,
    KEY_TEXT_TOKENS, KEY_TEXT_PADDING_MASK,
    forward_encoders, forward_fusion, forward_helper,
    gen_model, gen_alt_fusion_with_grad_mod,
)

from dataset.dataset import DATASET_HAS_TEXT_LIST

from common import (
    MAIN_DEVICE_KEY, BasicTrainer,
    update_arg, forward_fusion, MAIN_DEVICE_KEY, gen_model,
)

from utils import print_args, set_save_path, TeeOutput, printDebugInfo, setup_seed
from metrics import performanceMetric


def _uncertainty_weights(logits_list: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Compute per-sample importance weights from MLA eq. 7-8.

    For each output stream m, compute entropy  e_m = -p_m^T log(p_m).
    Then  λ_m = softmax(max_e - e_m)  so the most *confident* stream gets
    the highest weight on each sample.

    Args:
        logits_list: list of tensors, each shape (B, C).

    Returns:
        List of weight tensors, each shape (B, 1), summing to 1 across the
        list for each sample.
    """
    softmax = torch.nn.functional.softmax
    # (B, C) → (B,) entropy for each stream
    entropies = []
    for logits in logits_list:
        p = softmax(logits.detach(), dim=1).clamp(min=1e-8)
        e = -(p * p.log()).sum(dim=1)   # (B,)
        entropies.append(e)

    # stack → (B, M)
    E = torch.stack(entropies, dim=1)                   # (B, M)
    max_e = E.max(dim=1, keepdim=True).values           # (B, 1)
    # softmax over (max_e - e_m) so lower entropy → higher weight
    weights = torch.softmax(max_e - E, dim=1)           # (B, M)

    # split back to a list of (B, 1) tensors
    return [weights[:, i:i+1] for i in range(weights.shape[1])]


class AMST_F_Trainer(BasicTrainer):
    def __init__(self, args_str=None):
        self.parser = self.init_parser()
        self.args = self.init_parser().parse_args(args_str)
        self.init_logging()
        print_args(self.args)
        self.init_multi_gpu_env()
        self.init_env()
        self.model, self.device_map = self.init_model()
        self.modality_name_list = list(self.model["A"][KEY_ENCODERS].keys())
        self.modality_name_list.sort()

        self.m_skip_factor_map = {
            M_AUDIO_NAME: self.args.a_skip_factor,
            M_VISUAL_NAME: self.args.v_skip_factor,
            M_TEXT_NAME:   self.args.t_skip_factor,
        }

        self.skip_list_map = {}

        try:
            self.init_dataloader(self.args.using_ploader)
            self.init_optimizer_scheduler()
        except Exception as e:
            print(f"Error in init_dataloader: {e}")
            self.release()
            raise e

        self.softmax    = nn.Softmax(dim=1)
        self.criterion  = nn.CrossEntropyLoss()
        self.train_val_epoch_time_list = []

    # ------------------------------------------------------------------
    def init_parser(self):
        parser = super().init_parser()
        update_arg(parser, '--prefix', default='AMST-FULL', type=str,
                   help='prefix for the save path')
        parser.add_argument('--a_skip_factor', default=1, type=int,
                            help='skip factor for audio')
        parser.add_argument('--v_skip_factor', default=1, type=int,
                            help='skip factor for visual')
        parser.add_argument('--t_skip_factor', default=1, type=int,
                            help='skip factor for text')
        return parser

    # ------------------------------------------------------------------
    # ① Shared encoders
    #
    # Build one model (alt) with full encoder + fusion + helpers.
    # Build a second model (joint) with its own fusion + helpers, but
    # replace its KEY_ENCODERS with a *reference* to the alt encoders.
    # Optimisers then only cover shared encoders once (via the alt model).
    # ------------------------------------------------------------------
    def init_model(self):
        device_map = {
            M_TEXT_NAME:    0,
            M_AUDIO_NAME:   0,
            M_VISUAL_NAME:  0,
            MAIN_DEVICE_KEY: 0,
        }

        # --- alternating model (msum head, with gradient modification) ---
        self.args.fusion_method = 'msum'
        alt_model = gen_model(self.args)

        # Determine embedding dim from the alt model's fusion head input size.
        embedding_dim = alt_model[KEY_FUSION].fc.in_features

        # Replace the plain MLASum with one that carries a GradModifier.
        alt_model[KEY_FUSION] = gen_alt_fusion_with_grad_mod(
            self.args, embedding_dim)

        # --- joint model (concat head, no gradient modification) ---
        self.args.fusion_method = 'concat'
        joint_model = gen_model(self.args)

        # ① Replace joint encoders with shared references to alt encoders.
        #    nn.ModuleDict accepts assignment of existing modules; PyTorch
        #    will not double-register parameters when the same module object
        #    is referenced from two places in the same parent ModuleDict.
        joint_model[KEY_ENCODERS] = alt_model[KEY_ENCODERS]

        model = nn.ModuleDict()
        model["A"] = alt_model
        model["J"] = joint_model

        print(model,
              file=open(os.path.join(self.save_path, 'model.txt'), 'w'))

        model.to(device_map[MAIN_DEVICE_KEY])
        return model, device_map

    # ------------------------------------------------------------------
    def init_optimizer_scheduler(self):
        """
        Separate optimisers for each model part.

        Because the encoders are shared, we only create ONE encoder
        optimiser (for the alt model).  The joint optimiser map uses the
        same object so both backward passes update the same weights.
        """

        def _make_opt_sched(params):
            opt = optim.SGD(params,
                            lr=self.args.learning_rate,
                            momentum=0.9,
                            weight_decay=1e-4)
            sch = optim.lr_scheduler.StepLR(
                opt, self.args.lr_decay_step, self.args.lr_decay_ratio)
            return opt, sch

        # Shared encoders — one optimiser covers both modules.
        enc_opt, enc_sch = _make_opt_sched(
            self.model["A"][KEY_ENCODERS].parameters())

        # Alt-specific: fusion head + helpers.
        a_fusion_opt, a_fusion_sch = _make_opt_sched(
            self.model["A"][KEY_FUSION].parameters())
        a_helper_opt, a_helper_sch = _make_opt_sched(
            self.model["A"][KEY_HELPERS].parameters())

        # Joint-specific: fusion head + helpers.
        j_fusion_opt, j_fusion_sch = _make_opt_sched(
            self.model["J"][KEY_FUSION].parameters())
        j_helper_opt, j_helper_sch = _make_opt_sched(
            self.model["J"][KEY_HELPERS].parameters())

        # Expose maps that mirror the original structure so train_epoch can
        # use them consistently.
        self.A_optimizer_map = {
            KEY_ENCODERS: enc_opt,      # shared
            KEY_FUSION:   a_fusion_opt,
            KEY_HELPERS:  a_helper_opt,
        }
        self.J_optimizer_map = {
            KEY_ENCODERS: enc_opt,      # same object — shared encoders
            KEY_FUSION:   j_fusion_opt,
            KEY_HELPERS:  j_helper_opt,
        }
        self.A_scheduler_map = {
            KEY_ENCODERS: enc_sch,
            KEY_FUSION:   a_fusion_sch,
            KEY_HELPERS:  a_helper_sch,
        }
        self.J_scheduler_map = {
            KEY_ENCODERS: enc_sch,      # same scheduler for shared encoders
            KEY_FUSION:   j_fusion_sch,
            KEY_HELPERS:  j_helper_sch,
        }

    # ------------------------------------------------------------------
    def before_train_epoch(self):
        for modality_name in self.modality_name_list:
            if self.epoch % int(self.m_skip_factor_map[modality_name]) != 0:
                if modality_name not in self.skip_list_map:
                    self.skip_list_map[modality_name] = []
                self.skip_list_map[modality_name].append(self.epoch)
                print(f"skip {modality_name} at epoch {self.epoch}")

    def after_summary(self):
        print("SKIP INFO:")
        for modality_name in self.skip_list_map.keys():
            skip_list = self.skip_list_map.get(modality_name)
            print(f"{modality_name} skip {len(skip_list)} times:\n{skip_list}")
        print("END SKIP INFO")

    # ------------------------------------------------------------------
    def reinitialize_metrics(self):
        self.m_map = {}
        self.m_map["f"]     = performanceMetric(self.n_classes, name="f")
        self.m_map["alt"]   = performanceMetric(self.n_classes, name="alt")
        self.m_map["joint"] = performanceMetric(self.n_classes, name="joint")

        for modality_name in self.modality_name_list:
            self.m_map[f"alt_{modality_name}"] = performanceMetric(
                self.n_classes, f"alt_{modality_name}")
            self.m_map[f"joint_{modality_name}"] = performanceMetric(
                self.n_classes, f"joint_{modality_name}")

        self.m_h_map = {}
        for modality_name in self.modality_name_list:
            self.m_h_map[f"alt_{modality_name}_h"] = performanceMetric(
                self.n_classes, f"alt_{modality_name}_h")
            self.m_h_map[f"joint_{modality_name}_h"] = performanceMetric(
                self.n_classes, f"joint_{modality_name}_h")

    # ------------------------------------------------------------------
    # ③ Uncertainty-based test-time fusion
    #
    # Replaces the original fixed weighted average with per-sample entropy
    # weights across alt-per-modality logits and the joint logit.
    # ------------------------------------------------------------------
    def forward(self, data_packet):
        device_map         = self.device_map
        modality_name_list = self.modality_name_list
        softmax            = self.softmax
        criterion          = self.criterion
        m_map              = self.m_map
        m_h_map            = self.m_h_map

        input_dict, labels, infos = \
            self.prepare_input_dict(self.args.dataset, data_packet)
        labels_device = labels.to(device_map[MAIN_DEVICE_KEY])

        # ① Both modules share the same encoder object so one forward pass
        #    suffices.  The resulting embedding is used by both fusion heads.
        shared_embedding_dict = forward_encoders(
            self.model["A"][KEY_ENCODERS], input_dict)

        # Move to main device (needed for multi-GPU; no-op on single GPU).
        local_embedding_dict = {
            m: shared_embedding_dict[m].detach().to(device_map[MAIN_DEVICE_KEY])
            for m in modality_name_list
        }

        with torch.no_grad():
            # --- alt module outputs (one logit per modality) ---
            alt_out_f = forward_fusion(
                self.model["A"][KEY_FUSION], local_embedding_dict)

            # --- joint module output ---
            joint_out_f = forward_fusion(
                self.model["J"][KEY_FUSION], local_embedding_dict)

            # ③ Collect per-modality alt logits + joint logit and fuse
            #    with entropy-based weights.
            per_stream_logits = [
                self.model["A"][KEY_FUSION].get_out_m(m)
                for m in modality_name_list
            ] + [joint_out_f]

            weights = _uncertainty_weights(per_stream_logits)  # list of (B,1)

            final_out_f = sum(
                w * logit
                for w, logit in zip(weights, per_stream_logits)
            )

            # --- metrics ---
            def _update(metric_key, logits):
                if m_map is not None and metric_key in m_map:
                    pred = softmax(logits)
                    loss = criterion(logits, labels_device)
                    m_map[metric_key].update(pred, labels_device, loss=loss)

            _update("f",     final_out_f)
            _update("alt",   alt_out_f)
            _update("joint", joint_out_f)

            for modality_name in modality_name_list:
                out_alt_m = self.model["A"][KEY_FUSION].get_out_m(modality_name)
                _update(f"alt_{modality_name}",   out_alt_m)

                out_joint_m = self.model["J"][KEY_FUSION].get_out_m(modality_name)
                _update(f"joint_{modality_name}", out_joint_m)

        # --- helpers ---
        alt_helper_out_dict = forward_helper(
            self.model["A"][KEY_HELPERS], local_embedding_dict)
        joint_helper_out_dict = forward_helper(
            self.model["J"][KEY_HELPERS], local_embedding_dict)

        for modality_name in modality_name_list:
            def _update_h(metric_key, helper_logits):
                if m_h_map is not None and metric_key in m_h_map:
                    pred = softmax(helper_logits)
                    loss = criterion(helper_logits, labels_device)
                    m_h_map[metric_key].update(pred, labels_device, loss=loss)

            _update_h(f"alt_{modality_name}_h",
                      alt_helper_out_dict[modality_name])
            _update_h(f"joint_{modality_name}_h",
                      joint_helper_out_dict[modality_name])

        # Return the single shared embedding (used by both train methods).
        return (shared_embedding_dict,
                alt_helper_out_dict, joint_helper_out_dict,
                labels_device)

    # ------------------------------------------------------------------
    def valid(self, dataloader):
        self.reinitialize_metrics()
        self.model.eval()
        with torch.no_grad():
            for step, data_packet in enumerate(dataloader):
                self.forward(data_packet)

    # ------------------------------------------------------------------
    # ② Gradient-modified alternating training
    #
    # Compared to the original alt_train, two extra calls are added around
    # loss.backward():
    #   • fusion.prepare_grad_mod()  — registers the backward hook on the
    #     shared head before the backward pass.
    #   • fusion.finish_grad_mod()   — updates P after the optimizer step.
    # ------------------------------------------------------------------
    def alt_train(self, embedding_dict, labels_device, model, optimizer_map):
        fusion = model[KEY_FUSION]
        for modality_name in self.modality_name_list:
            fill_embed = None
            temp_input_dict = {
                name: (embedding_dict[name] if name == modality_name else fill_embed)
                for name in self.modality_name_list
            }

            out_m = forward_fusion(fusion, temp_input_dict)
            loss  = self.criterion(out_m, labels_device)

            if self.epoch % int(self.m_skip_factor_map[modality_name]) != 0:
                # Skipped epoch — still run forward (done above) so that the
                # gradient modifier can track the embedding, but do NOT update.
                # We still call finish_grad_mod so P stays current.
                fusion.finish_grad_mod()
                continue

            optimizer_map[KEY_FUSION].zero_grad()
            optimizer_map[KEY_ENCODERS].zero_grad()

            # ② Register backward hook before backward pass.
            fusion.prepare_grad_mod()

            loss.backward()
            optimizer_map[KEY_FUSION].step()
            optimizer_map[KEY_ENCODERS].step()

            # ② Update projection matrix P after the optimizer step.
            fusion.finish_grad_mod()

    # ------------------------------------------------------------------
    def joint_train(self, embedding_dict, labels_device, model, optimizer_map):
        temp_input_dict = {}
        fill_embed = None

        for modality_name in self.modality_name_list:
            if self.epoch % int(self.m_skip_factor_map[modality_name]) != 0:
                temp_input_dict[modality_name] = fill_embed
            else:
                temp_input_dict[modality_name] = embedding_dict[modality_name]

        if all(v is None for v in temp_input_dict.values()):
            return

        out_m = forward_fusion(model[KEY_FUSION], temp_input_dict)
        loss  = self.criterion(out_m, labels_device)

        optimizer_map[KEY_FUSION].zero_grad()
        optimizer_map[KEY_ENCODERS].zero_grad()
        loss.backward()
        optimizer_map[KEY_FUSION].step()
        optimizer_map[KEY_ENCODERS].step()

    # ------------------------------------------------------------------
    def train_epoch(self, dataloader):
        self.reinitialize_metrics()
        self.model.train()

        for step, data_packet in enumerate(dataloader):
            # --- initial forward (metrics + helpers) ---
            shared_embedding_dict, \
            alt_helper_out_dict, joint_helper_out_dict, \
            labels_device = self.forward(data_packet)

            # --- alternating module ---
            self.alt_train(shared_embedding_dict, labels_device,
                           self.model["A"], self.A_optimizer_map)

            # --- recompute embeddings for joint module ---
            # alt_train already stepped the shared encoders, so the old
            # embedding tensors are stale.  Run a fresh encoder forward.
            input_dict, _, _ = self.prepare_input_dict(
                self.args.dataset, data_packet)
            shared_embedding_dict = forward_encoders(
                self.model["A"][KEY_ENCODERS], input_dict)

            # --- joint module ---
            self.joint_train(shared_embedding_dict, labels_device,
                             self.model["J"], self.J_optimizer_map)

            # --- helpers (detached, so old embeddings are fine) ---
            self.A_optimizer_map[KEY_HELPERS].zero_grad()
            self.J_optimizer_map[KEY_HELPERS].zero_grad()
            for modality_name in self.modality_name_list:
                self.criterion(
                    alt_helper_out_dict[modality_name], labels_device
                ).backward()
                self.criterion(
                    joint_helper_out_dict[modality_name], labels_device
                ).backward()
            self.A_optimizer_map[KEY_HELPERS].step()
            self.J_optimizer_map[KEY_HELPERS].step()

        # Step schedulers once per epoch.
        # Encoder scheduler is shared, so stepping A's is sufficient.
        for key in [KEY_ENCODERS, KEY_FUSION, KEY_HELPERS]:
            self.A_scheduler_map[key].step()
        # Joint fusion + helpers have their own schedulers.
        self.J_scheduler_map[KEY_FUSION].step()
        self.J_scheduler_map[KEY_HELPERS].step()


if __name__ == "__main__":
    trainer = AMST_F_Trainer()
    trainer.train_validate()
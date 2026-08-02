# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

import warnings
import faiss
import os
import pandas as pd
import numpy as np
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.text_encoder.transformer
        self.positional_embedding = clip_model.text_encoder.positional_embedding
        self.ln_final = clip_model.text_encoder.ln_final
        self.text_projection = clip_model.text_encoder.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding[:-1, :]
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        row_indices = torch.arange(x.shape[0], device=x.device)
        eot_indices = tokenized_prompts.argmax(dim=-1).to(x.device)
        x = x[row_indices, eot_indices] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, prototype_number):
        super().__init__()
        device = next(clip_model.parameters()).device
        n_cls = len(classnames)
        n_ctx = prototype_number
        ctx_dim = clip_model.text_encoder.ln_final.weight.shape[0]
        tokenizer = clip_model.text_encoder.tokenizer

        ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]
        classnames = [prompt_prefix + " " + name for name in classnames]

        prompts = [name for name in classnames]
        tokenized_prompts = tokenizer(prompts).to(device)

        with torch.no_grad():
            _, embedding = clip_model.encode_text(tokenized_prompts)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    self.token_prefix,
                    ctx,
                    self.token_suffix,
                ],
                dim=1,
            )
            return prompts

        raise NotImplementedError(f"Unsupported class_token_position: {self.class_token_position}")


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


import types


def encode_text(self, input_ids: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Args:
        input_ids: torch.Tensor, shape (B, L)
    """
    input_ids = input_ids[:, :-1]
    text_latent, tokens = self.text_encoder(input_ids)
    if normalize:
        text_latent = F.normalize(text_latent, dim=-1)
    return text_latent, tokens


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class ClusterHead(nn.Module):
    def __init__(self, in_dim=768, num_clusters=10):
        super().__init__()
        self.num_clusters = num_clusters
        self.cluster_head_text = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_clusters),
            nn.Softmax(dim=1),
        )
        self.cluster_head_image = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_clusters),
            nn.Softmax(dim=1),
        )
        trunc_normal_(self.cluster_head_text[0].weight, std=0.02)
        trunc_normal_(self.cluster_head_text[2].weight, std=0.02)
        trunc_normal_(self.cluster_head_image[0].weight, std=0.02)
        trunc_normal_(self.cluster_head_image[2].weight, std=0.02)

    def forward(self, text, image):
        logit_text = self.cluster_head_text(text)
        logit_image = self.cluster_head_image(image)
        return logit_text, logit_image

class SlideKo(nn.Module):
    def __init__(
        self,
        input_size=768,
        text_prompt_path="resources/text_prompt/Claude/CAMELYON16_text_prompt.csv",
        prototype_number=16,
        num_clusters=10,
        num_concepts=10,
        num_heads=8,
        n_classes=2,
        titan_model_path=None,
        local_files_only=False,
        sc_loss_weight=0.1,
        ngsc_neighbors=5,
        concept_attention_temperature=0.1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.num_clusters = num_clusters
        self.num_concepts = num_concepts
        titan_source = titan_model_path or os.environ.get("TITAN_MODEL_PATH") or "MahmoodLab/TITAN"
        offline_env = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        local_only_env = os.environ.get("SLIDEKO_HF_LOCAL_FILES_ONLY") == "1"
        clip_model = AutoModel.from_pretrained(
            titan_source,
            trust_remote_code=True,
            local_files_only=bool(local_files_only or offline_env or local_only_env),
        )
        clip_model.encode_text = types.MethodType(encode_text, clip_model)

        _ = clip_model.eval()
        self.device = next(clip_model.parameters()).device
        clip_model = clip_model.to(self.device)

        self.feature_dim = clip_model.text_encoder.text_projection.shape[1]

        self.text_prompt = np.array(pd.read_csv(text_prompt_path)["Prompt"].values.tolist()).squeeze()
        self.prompt_learner = PromptLearner(self.text_prompt, clip_model.float(), prototype_number=prototype_number)
        self.text_encoder = TextEncoder(clip_model.float())

        self.Path_Adapter = Adapter(c_in=input_size, reduction=4)
        self.Text_Adapter = Adapter(c_in=input_size, reduction=4)

        self.ClusterHead = ClusterHead(in_dim=self.feature_dim, num_clusters=self.num_clusters)

        self.cross_attention0 = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.cross_attention1 = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.classifier = nn.Linear(self.feature_dim, n_classes)

        self.sc_loss_weight = float(sc_loss_weight)
        self.ngsc_neighbors = int(ngsc_neighbors)
        self.concept_attention_temperature = float(concept_attention_temperature)
        if self.concept_attention_temperature <= 0:
            raise ValueError("concept_attention_temperature must be greater than zero.")

    def get_concepts_refined(self, features, text_features, n_clu=20, topK=15):
        device = features.device
        features_np = features.cpu().detach().numpy().astype("float32")
        n_patches, d = features.shape

        k = min(n_clu, n_patches)
        kmeans = faiss.Kmeans(d=d, k=k, niter=20, min_points_per_centroid=5, verbose=False, gpu=True, seed=42)
        kmeans.train(features_np)
        cluster_centroids = torch.from_numpy(kmeans.centroids).to(device)

        centroids_norm = F.normalize(cluster_centroids, p=2, dim=1)
        text_features_norm = F.normalize(text_features, p=2, dim=1)

        sim_matrix = torch.matmul(centroids_norm, text_features_norm.T)
        softmax_text = torch.softmax(sim_matrix, dim=0)
        class_pred = torch.argmax(softmax_text, dim=0).long()

        selected_idx = torch.zeros_like(class_pred, dtype=torch.bool)
        for cluster_idx in range(n_clu):
            if (class_pred == cluster_idx).sum() == 0:
                continue
            class_index = torch.where(class_pred == cluster_idx)[0]
            softmax_class = softmax_text[:, class_index]
            confidence = softmax_class.max(dim=0)[0]
            rank = torch.argsort(confidence, descending=True)
            selected_idx[class_index[rank[:topK]]] = True
        selected_idx = selected_idx.cpu().numpy()

        text_embedding_selected = text_features_norm[selected_idx]
        feature_norm = F.normalize(features, p=2, dim=1)

        similarity = torch.matmul(feature_norm, text_embedding_selected.T)
        weights = torch.softmax(similarity / self.concept_attention_temperature, dim=1)
        retrieval_embedding = torch.matmul(weights, text_embedding_selected)
        retrieval_embedding = F.normalize(retrieval_embedding, dim=1)
        return retrieval_embedding

    def forward(self, **kwargs):
        data = kwargs["data"]
        h, p_text_features = data[0], data[1]
        h = h.clone().float().squeeze(0)
        p_text_features = p_text_features.clone().float().squeeze(0)

        prompts = self.prompt_learner()
        b_text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)

        features = self.Path_Adapter(h.float())
        p_text_features = self.Text_Adapter(p_text_features.float())

        # Slide-Adaptive Knowledge Instantiation (SAKI)
        p_text_features = self.get_concepts_refined(
            features,
            p_text_features,
            n_clu=self.num_clusters,
            topK=self.num_concepts,
        )

        compressed_features = features.unsqueeze(0)
        b_text_features = b_text_features.unsqueeze(0)
        p_text_features = p_text_features.unsqueeze(0)

        # Dual-Stream Cross-Attention (DSCA)
        visual_attended, _ = self.cross_attention0(
            b_text_features,
            compressed_features,
            compressed_features,
        )
        knowledge_attended, _ = self.cross_attention1(
            b_text_features,
            p_text_features,
            p_text_features,
        )
        fused_features = visual_attended.mean(1) + knowledge_attended.mean(1)
        logits = self.classifier(fused_features)

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        text_cluster_prob, image_cluster_prob = self.ClusterHead(
            p_text_features,
            compressed_features,
        )
        return {
            "logits": logits,
            "Y_prob": Y_prob,
            "Y_hat": Y_hat,
            "text_cluster_prob": text_cluster_prob,
            "image_cluster_prob": image_cluster_prob,
            "semantic_sc_features": p_text_features[0],
            "visual_sc_features": compressed_features[0],
        }

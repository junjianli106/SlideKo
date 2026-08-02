import importlib
import inspect
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from colorama import Fore, Style, init
from sklearn.metrics import (
    auc as calc_auc,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from MyLoss import create_loss
from MyOptimizer import create_optimizer


init(autoreset=True)


class ModelInterfaceCls(pl.LightningModule):

    # ---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterfaceCls, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        print(f"Loss function: {loss}")
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']

        self.training_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []

    @staticmethod
    def _neighbor_assignments(assignments, features, n_neighbors):
        n_samples = features.shape[0]
        if n_samples <= 1:
            return assignments

        k = max(1, min(int(n_neighbors), n_samples - 1))
        features = F.normalize(features.detach(), dim=-1)
        sim = features @ features.T
        sim.fill_diagonal_(-float("inf"))
        neighbor_idx = torch.topk(sim, k=k, dim=1).indices
        return assignments[neighbor_idx].mean(dim=1)

    @staticmethod
    def _ngsc_direction_loss(anchor_assignments, target_assignments, temperature):
        tau = max(float(temperature), 1e-8)
        anchor_cols = F.normalize(anchor_assignments.T, dim=1)
        target_cols = F.normalize(target_assignments.T, dim=1)

        cross_logits = anchor_cols @ target_cols.T / tau
        self_logits = anchor_cols @ anchor_cols.T / tau
        self_mask = torch.eye(self_logits.shape[0], dtype=torch.bool, device=self_logits.device)
        self_logits = self_logits.masked_fill(self_mask, -float("inf"))

        numerator = torch.diagonal(cross_logits)
        denominator = torch.logsumexp(torch.cat([cross_logits, self_logits], dim=1), dim=1)
        return -(numerator - denominator).mean()

    def _ngsc_loss(self, q_t, q_v, semantic_features, visual_features):
        """Neighborhood-guided semantic consistency loss (Eq. 15)."""
        n_neighbors = getattr(self.model, "ngsc_neighbors", 5)
        temperature = 0.5

        visual_neighbor_assign = self._neighbor_assignments(q_v, visual_features, n_neighbors)
        semantic_neighbor_assign = self._neighbor_assignments(q_t, semantic_features, n_neighbors)

        direct_cross_modal = self._ngsc_direction_loss(q_v, q_t, temperature)  # L_c(P_v, P_k)
        visual_to_semantic = self._ngsc_direction_loss(q_t, visual_neighbor_assign, temperature)  # L_c(P_k, A_v)
        semantic_to_visual = self._ngsc_direction_loss(q_v, semantic_neighbor_assign, temperature)  # L_c(P_v, A_k)
        return direct_cross_modal + visual_to_semantic + semantic_to_visual

    def training_step(self, batch, batch_idx):
        # ---->inference
        data, label = batch
        results_dict = self.model(data=data, is_train=True, label=label)

        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        q_t = results_dict['text_cluster_prob'].squeeze(0)   # (N, M)
        q_v = results_dict['image_cluster_prob'].squeeze(0)  # (N, M)
        loss_ngsc = self._ngsc_loss(
            q_t,
            q_v,
            results_dict['semantic_sc_features'],
            results_dict['visual_sc_features'],
        )
        loss = self.loss(logits, label) + self.model.sc_loss_weight * loss_ngsc

        self.training_step_outputs.append({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label})
        return loss

    def on_train_epoch_end(self):
        current_epoch = self.current_epoch + 1 # Assuming current_epoch is 0-indexed
        training_step_outputs = self.training_step_outputs

        logits = torch.cat([x['logits'] for x in training_step_outputs], dim=0).cpu().detach()
        all_probs = torch.cat([x['Y_prob'] for x in training_step_outputs], dim=0).cpu().detach()
        all_pred = torch.stack([x['Y_hat'].squeeze() for x in training_step_outputs]).cpu().detach()
        all_labels = torch.stack([x['label'] for x in training_step_outputs], dim=0).cpu().detach()

        # Standard Accuracy (optional)
        train_acc_standard = torch.sum(all_pred == all_labels.squeeze()).item() / len(all_labels)

        # Balanced Accuracy
        # Ensure labels and predictions are 1D NumPy arrays
        y_true_np = all_labels.squeeze().numpy()
        y_pred_np = all_pred.numpy()
        train_balanced_acc = balanced_accuracy_score(y_true_np, y_pred_np)

        train_f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        train_precision = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        train_recall = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)

        if self.n_classes == 2:
            # Ensure y_true_np is 1D for roc_auc_score if it's binary
            auc = roc_auc_score(y_true_np, all_probs[:, 1].numpy())
            aucs = [] # Not used in binary case, but kept for consistency
        else:
            aucs = []
            # Ensure y_true_np is 1D for label_binarize
            binary_labels = label_binarize(y_true_np, classes=[i for i in range(self.n_classes)])
            for class_idx in range(self.n_classes):
                # Check if the class is present in true labels
                if np.sum(binary_labels[:, class_idx]) > 0: # Check if class has any true samples
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx].numpy())
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan')) # Or handle as 0 or skip

            auc = np.nanmean(np.array(aucs)) if aucs else float('nan')


        # Beautified output
        print(Fore.CYAN + "═" * 80)
        print(Fore.GREEN + f"【TRAINING RESULTS】- Epoch: {Fore.YELLOW}{current_epoch}{Style.RESET_ALL}")
        print(f"  Accuracy: {Fore.YELLOW}{train_acc_standard:.4f}{Style.RESET_ALL}")
        print(f"  Balanced Accuracy:   {Fore.BLUE}{train_balanced_acc:.4f}{Style.RESET_ALL}") # Added
        print(f"  F1 Score:            {Fore.YELLOW}{train_f1:.4f}{Style.RESET_ALL}")
        print(f"  Precision:           {Fore.YELLOW}{train_precision:.4f}{Style.RESET_ALL}")
        print(f"  Recall:              {Fore.YELLOW}{train_recall:.4f}{Style.RESET_ALL}")
        print(f"  AUC:                 {Fore.YELLOW}{auc:.4f}{Style.RESET_ALL}")
        print(Fore.CYAN + "═" * 80)

        # Optional: Log training metrics (if you have a self.log method like in PyTorch Lightning)
        # self.log('train_acc_standard', train_acc_standard, on_epoch=True, logger=True)
        # self.log('train_balanced_acc', train_balanced_acc, on_epoch=True, logger=True)
        # self.log('train_f1', train_f1, on_epoch=True, logger=True)
        # self.log('train_precision', train_precision, on_epoch=True, logger=True)
        # self.log('train_recall', train_recall, on_epoch=True, logger=True)
        # self.log('train_auc', auc, on_epoch=True, logger=True)

        self.training_step_outputs = [] # Clear the outputs for the next training epoch


    def validation_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']
        self.val_step_outputs.append({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label})

    def on_validation_epoch_end(self):
        current_epoch = self.current_epoch + 1 # Assuming current_epoch is 0-indexed
        val_step_outputs = self.val_step_outputs

        logits = torch.cat([x['logits'] for x in val_step_outputs], dim=0).cpu().detach()
        all_probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim=0).cpu().detach()
        all_pred = torch.stack([x['Y_hat'].squeeze() for x in val_step_outputs]).cpu().detach()
        all_labels = torch.stack([x['label'] for x in val_step_outputs], dim=0).cpu().detach()

        # Standard Accuracy (optional)
        val_acc_standard = torch.sum(all_pred == all_labels.squeeze()).item() / len(all_labels)

        # Balanced Accuracy
        y_true_np = all_labels.squeeze().numpy()
        y_pred_np = all_pred.numpy()
        val_balanced_acc = balanced_accuracy_score(y_true_np, y_pred_np)

        val_f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        val_precision = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        val_recall = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)

        if self.n_classes == 2:
            auc = roc_auc_score(y_true_np, all_probs[:, 1].numpy())
            aucs = [] # Not used in binary case, but kept for consistency
        else:
            aucs = []
            # Ensure y_true_np is 1D for label_binarize
            binary_labels = label_binarize(y_true_np, classes=[i for i in range(self.n_classes)])
            for class_idx in range(self.n_classes):
                # Check if the class is present in true labels
                if np.sum(binary_labels[:, class_idx]) > 0:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx].numpy())
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan')) # Or handle as 0 or skip

            auc = np.nanmean(np.array(aucs)) if aucs else float('nan')


        # Beautified output
        print(Fore.CYAN + "═" * 80)
        print(Fore.MAGENTA + f"【VALIDATION RESULTS】- Epoch: {Fore.YELLOW}{current_epoch}{Style.RESET_ALL}")
        print(f"  Accuracy: {Fore.YELLOW}{val_acc_standard:.4f}{Style.RESET_ALL}")
        print(f"  Balanced Accuracy:   {Fore.GREEN}{val_balanced_acc:.4f}{Style.RESET_ALL}") # Added
        print(f"  F1 Score:            {Fore.YELLOW}{val_f1:.4f}{Style.RESET_ALL}")
        print(f"  Precision:           {Fore.YELLOW}{val_precision:.4f}{Style.RESET_ALL}")
        print(f"  Recall:              {Fore.YELLOW}{val_recall:.4f}{Style.RESET_ALL}")
        print(f"  AUC:                 {Fore.YELLOW}{auc:.4f}{Style.RESET_ALL}")
        print(Fore.CYAN + "═" * 80)

        # Log validation metrics
        self.log('val_f1', val_f1, prog_bar=True, on_epoch=True, logger=True)
        self.log('val_acc', val_acc_standard, prog_bar=True, on_epoch=True, logger=True) # Renamed for clarity
        self.log('val_balanced_acc', val_balanced_acc, prog_bar=True, on_epoch=True, logger=True) # Added
        self.log('val_precision', val_precision, on_epoch=True, logger=True)
        self.log('val_recall', val_recall, on_epoch=True, logger=True)
        self.log('val_auc', auc, on_epoch=True, logger=True)

        self.val_step_outputs = [] # Clear the outputs for the next validation epoch


    def test_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        self.test_step_outputs.append({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label})

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            print(Fore.YELLOW + "No test outputs to process." + Style.RESET_ALL)
            return

        test_step_outputs = self.test_step_outputs
        all_logits = torch.cat([x['logits'] for x in test_step_outputs], dim=0).cpu().detach()
        all_probs = torch.cat([x['Y_prob'] for x in test_step_outputs], dim=0).cpu().detach()
        # Ensure Y_hat is consistently shaped, especially if it comes from different batch sizes
        # Assuming Y_hat might be [batch_size, 1] or [batch_size], we use squeeze() and then stack.
        # If Y_hat is already consistently [batch_size], then stack is fine.
        # Let's adjust to handle potential single-element tensors that become 0-dim after squeeze.
        processed_Y_hat = []
        for x in test_step_outputs:
            y_hat_squeezed = x['Y_hat'].squeeze()
            if y_hat_squeezed.ndim == 0:  # If it became a 0-dim tensor
                y_hat_squeezed = y_hat_squeezed.unsqueeze(0)  # Make it 1-dim
            processed_Y_hat.append(y_hat_squeezed)
        all_pred = torch.cat(processed_Y_hat, dim=0).cpu().detach()

        # Ensure all_labels is consistently shaped. Assuming labels are [batch_size] or [batch_size, 1]
        processed_labels = []
        for x in test_step_outputs:
            label_squeezed = x['label'].squeeze()
            if label_squeezed.ndim == 0:  # If it became a 0-dim tensor
                label_squeezed = label_squeezed.unsqueeze(0)  # Make it 1-dim
            processed_labels.append(label_squeezed)
        all_labels = torch.cat(processed_labels, dim=0).cpu().detach()

        if all_pred.shape != all_labels.shape:
            print(
                Fore.RED + f"Shape mismatch! Predictions shape: {all_pred.shape}, Labels shape: {all_labels.shape}" + Style.RESET_ALL)
            # Attempt to squeeze labels if it's the common case of [N, 1] vs [N]
            if all_labels.ndim > 1 and all_labels.shape[1] == 1:
                all_labels_squeezed = all_labels.squeeze()
                if all_pred.shape == all_labels_squeezed.shape:
                    all_labels = all_labels_squeezed
                    print(Fore.YELLOW + f"Adjusted labels shape to: {all_labels.shape}" + Style.RESET_ALL)
                else:
                    print(
                        Fore.RED + "Could not resolve shape mismatch automatically. Please check your data." + Style.RESET_ALL)
                    return
            else:
                print(
                    Fore.RED + "Could not resolve shape mismatch automatically. Please check your data." + Style.RESET_ALL)
                return

        test_acc = torch.sum(all_pred == all_labels).item() / len(all_labels)

        # Balanced Accuracy
        test_balanced_acc = balanced_accuracy_score(all_labels.numpy(), all_pred.numpy())

        test_f1 = f1_score(all_labels.numpy(), all_pred.numpy(), average='macro', zero_division=0)
        test_precision = precision_score(all_labels.numpy(), all_pred.numpy(), average='macro', zero_division=0)
        test_recall = recall_score(all_labels.numpy(), all_pred.numpy(), average='macro', zero_division=0)

        if self.n_classes == 2:
            auc = roc_auc_score(all_labels.numpy(), all_probs[:, 1].numpy())
            aucs = []
        else:
            aucs = []
            binary_labels = label_binarize(all_labels.numpy(), classes=[i for i in range(self.n_classes)])
            for class_idx in range(self.n_classes):
                if np.sum(binary_labels[:, class_idx]) > 0 and binary_labels.shape[
                    1] > class_idx:  # Check if class is present and index is valid
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx].numpy())
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))

            auc = np.nanmean(np.array(aucs)) if aucs else float('nan')

        # --- 计算并打印每个类别的准确率 ---
        print(Fore.CYAN + "─" * 30 + " PER-CLASS ACCURACY " + "─" * 30 + Style.RESET_ALL)
        unique_labels = torch.unique(all_labels)
        class_accuracies = {}

        for label_class in unique_labels:
            class_idx = (all_labels == label_class)
            class_preds = all_pred[class_idx]
            class_true_labels = all_labels[class_idx]

            correct_predictions = torch.sum(class_preds == class_true_labels).item()
            total_instances = len(class_true_labels)

            if total_instances > 0:
                accuracy_for_class = correct_predictions / total_instances
                class_accuracies[label_class.item()] = accuracy_for_class
                print(
                    f"  Accuracy for Class {label_class.item()}: {Fore.MAGENTA}{accuracy_for_class:.4f}{Style.RESET_ALL} ({correct_predictions}/{total_instances})")
            else:
                class_accuracies[label_class.item()] = float('nan')  # 或者 0，根据你的偏好
                print(
                    f"  Accuracy for Class {label_class.item()}: {Fore.MAGENTA}N/A{Style.RESET_ALL} (No instances found)")
        # --- 结束每个类别准确率的计算和打印 ---

        print(Fore.CYAN + "═" * 80)
        print(Fore.RED + "【TEST RESULTS】" + Style.RESET_ALL)
        print(f"  Accuracy:    {Fore.YELLOW}{test_acc:.4f}{Style.RESET_ALL}")
        print(f"  Balanced Accuracy:   {Fore.GREEN}{test_balanced_acc:.4f}{Style.RESET_ALL}")
        print(f"  F1 Score:    {Fore.YELLOW}{test_f1:.4f}{Style.RESET_ALL}")
        print(f"  Precision:   {Fore.YELLOW}{test_precision:.4f}{Style.RESET_ALL}")
        print(f"  Recall:      {Fore.YELLOW}{test_recall:.4f}{Style.RESET_ALL}")
        print(f"  AUC:     {Fore.YELLOW}{auc:.4f}{Style.RESET_ALL}")
        print(
            f"  Data Shapes: {Fore.BLUE}logits {all_logits.shape}, labels {all_labels.shape}, predictions {all_pred.shape}{Style.RESET_ALL}")
        print(Fore.CYAN + "═" * 80)

        print(Fore.GREEN + "Saving results to: " + Style.RESET_ALL + f"{self.log_path}")

        metrics = {
            'acc': [test_acc],
            'balanced_acc': [test_balanced_acc],
            'f1': [test_f1],
            'precision': [test_precision],
            'recall': [test_recall],
            'auc': [auc]
        }
        # 添加每个类别的准确率到 metrics 字典
        for class_val, acc in class_accuracies.items():
            metrics[f'class_{class_val}_acc'] = [acc]

        df = pd.DataFrame(metrics)
        df.to_csv(os.path.join(self.log_path, 'test_metrics.csv'), index=False)

        probs_df = pd.DataFrame(all_probs.numpy())
        probs_df.to_csv(os.path.join(self.log_path, 'all_probs.csv'), index=False)

        labels_df = pd.DataFrame(all_labels.numpy())
        labels_df.to_csv(os.path.join(self.log_path, 'all_labels.csv'), index=False)

        logits_df = pd.DataFrame(all_logits.numpy())
        logits_df.to_csv(os.path.join(self.log_path, 'all_logits.csv'), index=False)

        np.savez_compressed(os.path.join(self.log_path, 'all_probs.npz'), all_probs.numpy())
        np.savez_compressed(os.path.join(self.log_path, 'all_labels.npz'), all_labels.numpy())
        np.savez_compressed(os.path.join(self.log_path, 'all_logits.npz'), all_logits.numpy())

        self.test_step_outputs = []

    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    def load_model(self):
        name = self.hparams.model.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
            print(Fore.GREEN + f"Model loaded successfully: {Fore.CYAN}{camel_name}")
        except:
            raise ValueError(Fore.RED + 'Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)



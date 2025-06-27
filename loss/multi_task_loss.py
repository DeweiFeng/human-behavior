import torch
import torch.nn as nn

# CCC Loss (used for affective regression)
class ConcordanceCCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_true_mean = torch.mean(y_true)
        y_pred_mean = torch.mean(y_pred)
        covariance = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
        y_true_var = torch.var(y_true)
        y_pred_var = torch.var(y_pred)
        ccc = (2 * covariance) / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2 + 1e-8)
        return 1 - ccc


# Multi-task Loss combining affective, pathology, social, cognitive, personality
class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_affective=1.0, lambda_pathology=1.0, lambda_social=1.0,
                 lambda_cognitive=1.0, lambda_personality=1.0):
        super().__init__()
        self.lambda_affective = lambda_affective
        self.lambda_pathology = lambda_pathology
        self.lambda_social = lambda_social
        self.lambda_cognitive = lambda_cognitive
        self.lambda_personality = lambda_personality

        self.ccc_loss = ConcordanceCCCLoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, outputs, targets):
        total_loss = 0.0

        if 'affective' in outputs:
            valence_loss = self.ccc_loss(outputs['affective'][:, 0], targets['affective'][:, 0])
            arousal_loss = self.ccc_loss(outputs['affective'][:, 1], targets['affective'][:, 1])
            total_loss += self.lambda_affective * (valence_loss + arousal_loss)

        if 'pathology' in outputs:
            pathology_loss = self.mse_loss(outputs['pathology'], targets['pathology'])
            total_loss += self.lambda_pathology * pathology_loss

        if 'social' in outputs:
            social_loss = self.bce_loss(outputs['social'], targets['social'])
            total_loss += self.lambda_social * social_loss

        if 'cognitive' in outputs:
            cognitive_loss = self.mse_loss(outputs['cognitive'], targets['cognitive'])
            total_loss += self.lambda_cognitive * cognitive_loss

        if 'personality' in outputs:
            personality_loss = self.bce_loss(outputs['personality'], targets['personality'])
            total_loss += self.lambda_personality * personality_loss

        return total_loss
    

def multi_task_loss(preds, labels, task):
    if task == "phq8":
        return sum(nn.CrossEntropyLoss()(logits, labels[i]) for i, logits in enumerate(preds)) / 8
    elif task == "phq_binary":
        return nn.CrossEntropyLoss()(preds, labels)
    elif task == "emotion":
        return nn.CrossEntropyLoss()(preds, labels)
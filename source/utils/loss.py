import torch


class BCEWithLogitsLossL2(torch.nn.Module):
    def __init__(self, model, lambda_l2=0.00):
        super(BCEWithLogitsLossL2, self).__init__()
        self.model = model
        self.lambda_l2 = lambda_l2
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        # Compute the original BCE Loss
        loss = self.bce_loss(output, target)

        # Compute L2 regularization term for the two linear layers
        l2_reg = torch.tensor(0., requires_grad=True).to(output.device)
        for name, param in self.model.named_parameters():
            if 'lincomb' in name or 'lin' in name:
                l2_reg = l2_reg + torch.norm(param)

        # Add L2 regularization term to the loss
        loss = loss + self.lambda_l2 * l2_reg

        return loss

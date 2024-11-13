import torch

class ConsistentLoss():
    def __init__(self, mu_off=0, mu_pos=0, mu_ortho=0, normalized=True, A_ortho=True):
        self.mu_off = mu_off
        self.mu_pos = mu_pos
        self.mu_ortho = mu_ortho
        self.normalized = normalized
        self.A_ortho = A_ortho

        self.loss_functions = [
            (self.mu_off, self.off_penalty_loss, ["consistent_bases", "Ls", "eVals"]),
            (self.mu_pos, self.pos_contrastive_loss, ["descriptors", "Ms", "consistent_bases"]),
            (self.mu_ortho, self.ortho_loss, ["consistent_bases", "Ms"])
        ]

    def loss(self, eVals, consistent_bases, Ls, Ms, descriptors):
        final_loss = 0
        loss_details = {}

        parameters = {
            'eVals': eVals, # (B, 2, K)
            'consistent_bases': consistent_bases, # (B, 2, N, K)
            'Ls': Ls, # (B, 2, N, N)
            'Ms': Ms, # (B, 2, N)
            'descriptors': descriptors # (B, 2, N, D)
        }

        for weight, loss_fn, required_params in self.loss_functions:
            if weight != 0:
                loss_value = loss_fn(**{param: parameters[param] for param in required_params}) * weight
                final_loss += loss_value
                loss_details[loss_fn.__name__] = loss_value

        return final_loss, loss_details

    def off_penalty_loss(self, consistent_bases, Ls, eVals):

        PhisT = torch.transpose(consistent_bases, 2, 3) # (B, 2, N, K) -> (B, 2, K, N)
        PhisTL = torch.matmul(PhisT, Ls) # (B, 2, K, N) x (B, 2, N, N) -> (B, 2, K, N)
        PhisTLPhis = torch.matmul(PhisTL, consistent_bases) # (B, 2, K, N) x (B, 2, N, K) -> (B, 2, K, K)
        eVals = torch.diag_embed(eVals) # (B, 2, K) -> (B, 2, K, K)
        off_penalty_loss = torch.linalg.matrix_norm(PhisTLPhis - eVals).sum()
        return off_penalty_loss

    def pos_contrastive_loss(self, descriptors, Ms, consistent_bases):

        diff = self._frequency_domain_projection(descriptors[:, 0, :, :], Ms[:, 0, :], consistent_bases[:, 0, :, :]) \
             - self._frequency_domain_projection(descriptors[:, 1, :, :], Ms[:, 1, :], consistent_bases[:, 1, :, :])
        pos_contrastive_loss = torch.linalg.matrix_norm(diff).sum()
        return pos_contrastive_loss

    def ortho_loss(self, consistent_bases, Ms):
        ortho_loss = 0
        k = consistent_bases.shape[3]
        I_k = torch.eye(k).to(consistent_bases.device)
        for b in range(2):
            Phi = consistent_bases[:, b, :, :]
            Phi_transpose = Phi.transpose(1, 2)
            M = Ms[:, b, :]
            M = torch.diag_embed(M)
            ortho_loss += torch.linalg.matrix_norm(Phi_transpose @ M @ Phi - I_k).sum()
        return ortho_loss

    def _frequency_domain_projection(self, descriptors, Ms, consistent_bases):
        if self.A_ortho:
            Ms = torch.diag_embed(Ms) # (B, N) -> (B, N, N)
            weighted_cb = torch.matmul(Ms, consistent_bases) # (B, N, N) x (B, N, K) -> (B, N, K) 
            descriptorsT = torch.transpose(descriptors, 1, 2) # (B, N, D) -> (B, D, N)
            results = torch.matmul(descriptorsT, weighted_cb) # (B, D, N) x (B, N, K) -> (B, D, K)
        else:
            descriptorsT = torch.transpose(descriptors, 1, 2) # (B, N, D) -> (B, D, N)
            results = torch.matmul(descriptorsT, consistent_bases) # (B, D, N) x (B, N, K) -> (B, D, K)
        return results

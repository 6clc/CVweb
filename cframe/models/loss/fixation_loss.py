import torch.nn as nn
import torch

eps = 1e-10


class KlDivergence(nn.Module):
    def __init__(self, config=None):
        super().__init__()

    def forward(self, data_dict):
        '''

        :param y_pred: saliency map
        :param y_true: groundturth density map
        :return: kl-div
        '''
        # 2torch.kl_div()
        y_pred = data_dict['pred']
        y_true = data_dict['fixation']
        assert y_pred.ndim == y_true.ndim and y_true.ndim == 3, 'dim of pred should be equal to target'
        max_y_pred = torch.max(
            torch.max(y_pred, dim=1, keepdim=True)[0],
            dim=2, keepdim=True
        )[0]
        y_pred = y_pred / max_y_pred

        sum_y_true = torch.sum(
            torch.sum(y_true, dim=1, keepdim=True),
            dim=2, keepdim=True)
        sum_y_pred = torch.sum(
            torch.sum(y_pred, dim=1, keepdim=True),
            dim=2, keepdim=True
        )

        y_true = y_true / (sum_y_true + eps)
        y_pred = y_pred / (sum_y_pred + eps)

        return 10 * torch.sum(
            y_true * torch.log((y_true / (y_pred + eps)) + eps)
        )


class CorrelationCoefficient(nn.Module):
    def __init__(self, config=None):
        super().__init__()

    def forward(self, data_dict):
        y_pred = data_dict['pred']
        y_true = data_dict['saliency']
        assert y_pred.ndim == y_true.ndim and y_true.ndim == 3, 'dim of pred should be equal to target'
        max_y_pred = torch.max(
            torch.max(y_pred, dim=1, keepdim=True)[0],
            dim=2, keepdim=True
        )[0]
        y_pred = y_pred / max_y_pred

        sum_y_true = torch.sum(
            torch.sum(y_true, dim=1, keepdim=True),
            dim=2, keepdim=True)
        sum_y_pred = torch.sum(
            torch.sum(y_pred, dim=1, keepdim=True),
            dim=2, keepdim=True
        )

        y_true = y_true / (sum_y_true + eps)
        y_pred = y_pred / (sum_y_pred + eps)

        B, H, W = y_pred.shape
        N = H * W

        sum_x = torch.sum(
            torch.sum(y_true, dim=1, keepdim=True),
            dim=2, keepdim=True
        )
        sum_y = torch.sum(
            torch.sum(y_pred, dim=1, keepdim=True),
            dim=2, keepdim=True
        )

        sum_prod = torch.sum(
            torch.sum(torch.mul(y_true, y_pred), dim=1, keepdim=True),
            dim=2, keepdim=True
        )

        sum_x_square = torch.sum(
            torch.sum(torch.mul(y_true, y_true), dim=1, keepdim=True),
            dim=2, keepdim=True
        )
        sum_y_square = torch.sum(
            torch.sum(torch.mul(y_pred, y_pred), dim=1, keepdim=True),
            dim=2, keepdim=True
        )

        num = sum_prod - ((sum_x * sum_y) / N)
        den = torch.sqrt((sum_x_square - torch.mul(sum_x, sum_x) / N) * (sum_y_square - torch.mul(sum_y, sum_y) / N))

        return torch.sum(-2 * num / den)


class NSS(nn.Module):
    def __init__(self, config=None):
        super().__init__()

    def forward(self, data_dict):
        y_pred = data_dict['pred']
        y_true = data_dict['saliency']
        assert y_pred.ndim == y_true.ndim and y_true.ndim == 3, 'dim of pred should be equal to target'
        max_y_pred = torch.max(
            torch.max(y_pred, dim=1, keepdim=True)[0],
            dim=2, keepdim=True
        )[0]
        y_pred = y_pred / max_y_pred
        y_pred_flatten = torch.flatten(y_pred, start_dim=1)

        y_mean = torch.mean(y_pred_flatten, dim=-1)
        y_std = torch.std(y_pred_flatten, dim=-1)

        # print(y_pred.shape, y_mean[:,None,None].shape)
        y_pred = (y_pred - y_mean[:, None, None]) / (y_std[:, None, None] + eps)

        return -torch.sum(torch.div(torch.sum(torch.sum(y_true * y_pred, dim=1, keepdim=True), dim=2, keepdim=True),
                                    torch.sum(torch.sum(y_true, dim=1, keepdim=True), dim=2, keepdim=True)))


class SamLoss(nn.Module):
    def __init__(self, config=None):
        super(SamLoss, self).__init__()
        print('use SamLoss')
        self.nss = NSS(config)
        self.kl = KlDivergence(config)
        self.cc = CorrelationCoefficient(config)

    def forward(self, data_dict):
        nss = self.nss(data_dict)
        kl = self.kl(data_dict)
        cc = self.cc(data_dict)
        return nss + kl + cc


if __name__ == '__main__':
    y_true1 = torch.randint(low=0, high=255, size=(2, 2))
    y_pred1 = torch.randint(low=0, high=255, size=(2, 2))

    y_true1 = y_true1.float()
    y_pred1 = y_pred1.float()

    y_pred = torch.stack([y_pred1, y_true1, y_true1], dim=0)
    y_true = torch.stack([y_true1, y_pred1, y_true1], dim=0)

    # y_pred = torch.stack([y_true1], dim=0)
    # y_true = torch.stack([y_true1], dim=0)

    kl = KlDivergence()
    kl_loss = kl(y_pred, y_true)
    print(kl_loss.item())
    cc = CorrelationCoefficient()
    cc_loss = cc(y_pred, y_true)
    print(cc_loss.item())

    nss = NSS()
    nss_loss = nss(y_pred, y_true)
    print(nss_loss)

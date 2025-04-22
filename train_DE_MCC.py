import argparse
from dataloader import load_data
from torch.utils.data import DataLoader
from loss import Loss
from evaluation import evaluate
from DE_MCC import DE_MCC
import torch
import numpy as np
import warnings
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--dataset', default='EEG')  # BS_DE
parser.add_argument("--feature_dim", default=256)
parser.add_argument("--epochs", default=120)
parser.add_argument("--view", type=int, default=6)
args = parser.parse_args()
if args.dataset == "mnist_mv":
    args.feature_dim = 288


class Optimizer:
    def __init__(self, params):
        self.clip_norm = 5.0
        self.scheduler_step_size = 50
        self.scheduler_gamma = 0.5
        self.params = params
        self._opt = torch.optim.Adam(params, lr=1e-3)
        if self.scheduler_step_size is not None:
            assert self.scheduler_gamma is not None
            self._sch = torch.optim.lr_scheduler.StepLR(self._opt, step_size=self.scheduler_step_size,
                                                         gamma=self.scheduler_gamma)
        else:
            self._sch = None

    def zero_grad(self):
        return self._opt.zero_grad()

    def step(self, epoch):
        if self._sch is not None:
            if epoch.is_integer() and epoch > 0:
                self._sch.step()

        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_norm)

        return self._opt.step()


def set_model_gradients(model, enable_gate=True, enable_other=False):
    """
    Set gradients for different parts of the model based on the given flags.
    """
    for p in model.gate.parameters():
        p.requires_grad = enable_gate
    for p in model.old_model.parameters():
        p.requires_grad = enable_other
    for p in model.new_model.parameters():
        p.requires_grad = enable_other
    for p in model.single.parameters():
        p.requires_grad = enable_other
    for p in model.cluster_module.parameters():
        p.requires_grad = enable_other


def train_epoch(epoch, view, model, data_loader, criterion, optimizer, records, device, enable_gate):
    """
    Train the model for a single epoch with the specified gradient configuration.
    """
    tot_loss = 0.
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, output, hidden, fused = model(xs)
        loss, _ = criterion.forward_cluster(hidden, output, fused)
        loss.backward()
        optimizer.step(epoch - 1 + batch_idx / len(data_loader))
        tot_loss += loss.item()
    if epoch % 20 == 0:
        print(tot_loss)
    if epoch == args.epochs:
        records.append(tot_loss / len(data_loader))
    return tot_loss


def valid(model, device, dataset, total_view):
    loader = DataLoader(
        dataset,
        batch_size=100,
        shuffle=False,
    )
    model.eval()
    pred_vector = []
    labels_vector = []
    fuseds = []
    data = []
    for _, (xs, y, _) in enumerate(loader):
        data.extend(xs[0].numpy())
        for v in range(total_view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            _, _, output, _, fused = model(xs)
            fuseds.extend(fused[2].detach().cpu().numpy())
            pred_vector.extend(output.detach().cpu().numpy())
        labels_vector.extend(y.numpy())

    labels = np.array(labels_vector).reshape(len(labels_vector))
    pred_vec = np.argmax(np.array(pred_vector), axis=1)
    ch = calinski_harabasz_score(fuseds, pred_vec)
    ss = silhouette_score(fuseds, pred_vec)
    dbi = davies_bouldin_score(fuseds, pred_vec)
    dict_ = {}
    for _ in pred_vec:
        if _ not in dict_:
            dict_[_] = 0
        else:
            dict_[_] += 1
    nmi, ari, acc, pur = evaluate(labels, pred_vec)
    return [acc, nmi, ari, pur], ch, ss


def main():
    seed = 2
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    view_num = args.view
    T = 10
    records = {"metrics": []}
    ch_list = []
    ss_list = []

    for t in range(T):
        model = DE_MCC(view_num - 1, view_num, dims, args.feature_dim, class_num)
        model = model.to(device)
        criterion = Loss(args.batch_size, class_num, device)

        for epoch in range(args.epochs):
            if (epoch // 20) % 2 == 0:
                set_model_gradients(model, enable_gate=False, enable_other=True)
                optimizer = Optimizer(filter(lambda p: p.requires_grad, model.parameters()))
                train_epoch(epoch + 1, view, model, data_loader, criterion, optimizer, [], device, enable_gate=False)
            else:
                set_model_gradients(model, enable_gate=True, enable_other=False)
                optimizer = Optimizer(filter(lambda p: p.requires_grad, model.parameters()))
                train_epoch(epoch + 1, view, model, data_loader, criterion, optimizer, [], device, enable_gate=True)

        res, ch, ss = valid(model, device, dataset, view)
        ch_list.append(ch)
        ss_list.append(ss)

        for i in range(4):
            if len(records["metrics"]) <= i:
                records["metrics"].append([])
            records["metrics"][i].append(res[i])

    ju_dic = {'ch': ch_list, 'ss': ss_list}
    df = pd.DataFrame(ju_dic)
    exp_name = 'view2'
    df.to_csv(f'out_{exp_name}.csv', index=False)
    ind_ = np.argmax(np.array(records["metrics"][0]))
    print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f}'.format(records["metrics"][0][ind_],
                                                        records["metrics"][1][ind_],
                                                        records["metrics"][3][ind_]))


if __name__ == '__main__':
    main()
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import CheckPoint
import Loader

features_path = 'features_data'
loss_weight = [1, 1, 1]


def multi_obj_loss(output: torch.tensor, targets: torch.tensor,
                   outputsreg: torch.tensor, targetsreg: torch.tensor,
                   outputobj: torch.tensor, objtarget: torch.tensor,
                   loss: torch.nn, device: torch.device):
    batch = outputobj.size(0)

    y = objtarget.reshape(batch, -1)
    x = outputobj.reshape(batch, -1)
    l3 = loss[2](x, y.to(torch.float).to(device))
    isobj = y.to(torch.bool).view(-1)

    y = targets.view(-1).to(device)
    x = output.permute(0, 2, 3, 1).contiguous().view(-1, 20)
    l1 = loss[0](x[isobj], y[isobj].long())
    # l1 += loss[3](x[isobj],y[isobj].long())

    predicted_targets = x[isobj].argmax(dim=1)
    correct = (predicted_targets == y[isobj]).sum().item()
    N = y[isobj].shape[0]

    y = targetsreg.permute(0, 2, 3, 1).to(device)
    x = outputsreg.permute(0, 2, 3, 1).to(device)
    l2 = []

    for i in range(4):
        pointy = y[:, :, :, i].contiguous().view(-1)
        pointx = x[:, :, :, i].contiguous().view(-1)
        l2.append(loss[1](pointx[isobj], pointy[isobj]))

    multi_loss = loss_weight[0] * l1 + loss_weight[2] * l3 + loss_weight[1] * (l2[0] + l2[1] + l2[2] + l2[3])

    return multi_loss, N, correct


def train(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
          f_loss: list, optimizer: torch.optim, device: torch.device, penalty: bool):
    model.train()

    N = 0
    all_loss, correct = 0.0, 0.0
    pbar = tqdm(total=len(loader), desc="Train batch : ")
    for e in loader:

        inputs, targetsreg, targets = e[0], e[1], e[2]

        outputs = model(inputs)
        outputsreg, output = outputs[0], outputs[1]

        # Multiple objects detection case
        if (len(outputs) > 2):
            objtarget, outputobj, targets = e[2], outputs[2], e[3]
            loss, n, c = multi_obj_loss(output, targets, outputsreg, targetsreg, outputobj, objtarget, f_loss, device)
            correct, N = correct + c, N + n

        else:
            loss = f_loss[0](output, targets.to(device)) + f_loss[1](outputsreg, targetsreg.to(device))
            N += inputs.shape[0]
            predicted_targets = output.argmax(dim=1)
            correct += (predicted_targets.to(device) == targets.to(device)).sum().item()

        all_loss += loss
        optimizer.zero_grad()

        loss.backward()

        if penalty:
            model.penalty().backward()

        optimizer.step()

        pbar.update(1)

    pbar.close()

    return all_loss / N, correct / N


def test(model: torch.nn.Module,
         loader: torch.utils.data.DataLoader,
         f_loss: list, device: torch.device):
    with torch.no_grad():

        model.eval()

        N = 0
        all_loss, correct = 0.0, 0.0,

        pbar = tqdm(total=len(loader), desc="Eval batch : ")
        for batch in loader:
            inputs, targetsreg, targets = batch[0], batch[1], batch[2]

            outputs = model(inputs)
            outputsreg, output = outputs[0], outputs[1]

            # Multiple objects detection case
            if (len(outputs) > 2):
                objtarget, outputobj, targets = batch[2], outputs[2], batch[3]
                loss, n, c = multi_obj_loss(output, targets, outputsreg, targetsreg, outputobj, objtarget, f_loss,
                                            device)
                correct, N = correct + c, N + n
            else:

                loss = loss_weight[0] * f_loss[0](output, targets.to(device)) + loss_weight[1] * f_loss[1](outputsreg,
                                                                                                           targetsreg.to(
                                                                                                               device))
                N += inputs.shape[0]
                predicted_targets = output.argmax(dim=1)
                correct += (predicted_targets.to(device) == targets.to(device)).sum().item()

            pbar.update(1)
            all_loss += loss
        pbar.close()
        return all_loss / N, correct / N


def train_nep_save(epochs: int, tag: str, f_loss: list,
                   optimizer: torch.optim, model, penalty=False):
    train_loader, valid_loader, test_loader = Loader.load_features(features_path)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)
    path = CheckPoint.save_path()
    model_checkpoint = CheckPoint.ModelCheckpoint(path + "/best_model.pt", model)

    tensorboard_writer = SummaryWriter(log_dir=path)

    for ep in range(epochs):
        print("Epoch {}".format(ep))
        train_reg_loss, train_acc = train(model, train_loader, f_loss, optimizer, device, penalty)

        val_reg_loss, val_acc = test(model, valid_loader, f_loss, device)
        print(" Validation : Loss : {:.4f}, Acc : {:.4f}".format(val_reg_loss, val_acc))

        model_checkpoint.update(val_reg_loss + (1.0 - val_acc))

        tensorboard_writer.add_scalars(tag + 'loss', {'metrics/train_regresion_loss': train_reg_loss,
                                                      'metrics/val_regresion_loss': val_reg_loss}, ep)
        tensorboard_writer.add_scalars(tag + 'acc', {'metrics/train_acc': train_acc, 'metrics/val_acc': val_acc}, ep)

    tensorboard_writer.close()

    model = torch.load(path + "/best_model.pt")
    loss, acc = test(model, test_loader, f_loss, device)
    print(" Test : Loss : {:.4f}, Acc : {:.4f}".format(loss, acc))

    file = open("results_acc.res", "a+")
    print(str(tag) + " " + str(acc), file=file)
    file.close()


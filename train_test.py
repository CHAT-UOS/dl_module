from tqdm import tqdm
import time


def train(model, epoch, criterion, optimizer, train_loader, device):

    model.train()
    train_loss = 0
    correct = 0

    with tqdm(total=len(train_loader), desc=f'Train Epoch [{epoch}]', ncols=100) as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(
                device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            # if batch_idx % 100 == 0 and batch_idx != 0:
            #    print(f'Train Epoch: {epoch} | Batch Status: {batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%) | Loss: {loss.item():.6f}')

            pbar.update(1)


def test(model, epoch, criterion, optimizer, test_loader, device):

    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    result = round((100. * correct / len(test_loader.dataset)).item(), 2)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.1f}%)')
    return epoch, result

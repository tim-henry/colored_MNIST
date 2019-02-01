from __future__ import print_function
import argparse
import colored_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2_number = nn.Linear(500, 10)
        self.fc2_color = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        num = self.fc2_number(x)
        col = self.fc2_color(x)
        return F.log_softmax(num, dim=1), F.log_softmax(col, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        num_target, col_target = target[:, 0], target[:, 1]
        optimizer.zero_grad()

        num_output, col_output = model(data)
        loss = F.nll_loss(num_output, num_target) + F.nll_loss(col_output, col_target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    num_correct_count = 0
    col_correct_count = 0
    correct_count = 0

    left_out_num_correct_count = 0
    left_out_col_correct_count = 0
    left_out_correct_count = 0
    left_out_count = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            num_target, col_target = target[:, 0], target[:, 1]

            num_output, col_output = model(data)
            test_loss += F.nll_loss(num_output, num_target, reduction='sum').item()  # sum up batch loss
            test_loss += F.nll_loss(col_output, col_target, reduction='sum').item()

            # Calculate accuracy
            # get the index of the max log-probability
            pred = torch.cat((num_output.argmax(dim=1, keepdim=True), col_output.argmax(dim=1, keepdim=True)), 1)
            num_correct, col_correct = pred.eq(target.view_as(pred))[:, 0], pred.eq(target.view_as(pred))[:, 1]
            correct = num_correct * col_correct  # both must be correct

            num_correct_count += num_correct.sum().item()
            col_correct_count += col_correct.sum().item()
            correct_count += correct.sum().item()

            # Calculate left-out accuracy
            left_out = colored_dataset.LeftOutColoredMNIST.left_out
            mask = torch.Tensor(((num_target.numpy() == left_out[0]) * (col_target.numpy() == left_out[1]))
                                .astype("uint8")).byte()

            left_out_num_correct = num_correct * mask
            left_out_col_correct = col_correct * mask
            left_out_correct = left_out_num_correct * left_out_col_correct

            left_out_num_correct_count += left_out_num_correct.sum().item()
            left_out_col_correct_count += left_out_col_correct.sum().item()
            left_out_correct_count += left_out_correct.sum().item()
            left_out_count += mask.sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          '(Number Accuracy: {}/{} ({:.0f}%), Color Accuracy: {}/{} ({:.0f}%))\n\n'
          'Left-Out Accuracy: {}/{} ({:.0f}%)\n'
          '(Left-Out Number Accuracy: {}/{} ({:.0f}%), Left-Out Color Accuracy: {}/{} ({:.0f}%))\n\n'.format(
            test_loss,
            correct_count, len(test_loader.dataset), 100. * correct_count / len(test_loader.dataset),
            num_correct_count, len(test_loader.dataset), 100. * num_correct_count / len(test_loader.dataset),
            col_correct_count, len(test_loader.dataset), 100. * col_correct_count / len(test_loader.dataset),
            left_out_correct_count, left_out_count, 100. * left_out_correct_count / left_out_count,
            left_out_num_correct_count, left_out_count, 100. * left_out_num_correct_count / left_out_count,
            left_out_col_correct_count, left_out_count, 100. * left_out_col_correct_count / left_out_count
            ))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        colored_dataset.LeftOutColoredMNIST('../data', train=True, download=True),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        colored_dataset.ColoredMNIST('../data', train=False),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()

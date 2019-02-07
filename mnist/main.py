from __future__ import print_function
import argparse
import colored_dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms


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


def test(args, model, device, test_loader, left_out):
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
            mask = np.zeros(num_target.size())
            for pair in left_out:
                diff_array = np.absolute(target.numpy() - np.array(pair))
                mask = np.logical_or(mask, diff_array.sum(axis=1) == 0)

            mask = torch.Tensor(mask.astype("uint8")).byte()

            left_out_num_correct = num_correct * mask
            left_out_col_correct = col_correct * mask
            left_out_correct = left_out_num_correct * left_out_col_correct

            left_out_num_correct_count += left_out_num_correct.sum().item()
            left_out_col_correct_count += left_out_col_correct.sum().item()
            left_out_correct_count += left_out_correct.sum().item()
            left_out_count += mask.sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          '(Number Accuracy: {}/{} ({:.0f}%), Color Accuracy: {}/{} ({:.0f}%))\n'.format(
        test_loss,
        correct_count, len(test_loader.dataset), 100. * correct_count / len(test_loader.dataset),
        num_correct_count, len(test_loader.dataset), 100. * num_correct_count / len(test_loader.dataset),
        col_correct_count, len(test_loader.dataset), 100. * col_correct_count / len(test_loader.dataset)
    ))

    left_out_acc = None
    if left_out_count > 0:
        print('Left-Out Accuracy: {}/{} ({:.0f}%)\n'
              '(Left-Out Number Accuracy: {}/{} ({:.0f}%), Left-Out Color Accuracy: {}/{} ({:.0f}%))\n'.format(
            left_out_correct_count, left_out_count, 100. * left_out_correct_count / left_out_count,
            left_out_num_correct_count, left_out_count, 100. * left_out_num_correct_count / left_out_count,
            left_out_col_correct_count, left_out_count, 100. * left_out_col_correct_count / left_out_count
        ))
        left_out_acc = left_out_correct_count / left_out_count

    return (correct_count / len(test_loader.dataset),
            left_out_acc,
            (correct_count - left_out_correct_count) / (len(test_loader.dataset) - left_out_count))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--jitter', type=float, default=0, metavar='J',
                        help='Color jitter (default=0')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    keep_pcts = [1, 0.8, 0.6, 0.4, 0.2]
    for keep_pct in keep_pcts:
        print("Keep pct: ", keep_pct)
        torch.manual_seed(args.seed)

        device = torch.device("cuda" if use_cuda else "cpu")

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        train_loader = torch.utils.data.DataLoader(
            colored_dataset.LeftOutColoredMNIST('../data', train=True, download=True,
                                                transform=transforms.ColorJitter(args.jitter, args.jitter, args.jitter, args.jitter),
                                                pct_to_keep=keep_pct),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            colored_dataset.LeftOutColoredMNIST('../data', train=True, download=True,
                                                transform=transforms.ColorJitter(args.jitter, args.jitter, args.jitter, args.jitter),
                                                pct_to_keep=1),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        test_accs, left_out_accs, non_left_out_accs = [], [], []
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test_acc, left_out_acc, non_left_out_acc = \
                test(args, model, device, test_loader, train_loader.dataset.left_out)
            test_accs.append(test_acc)
            non_left_out_accs.append(non_left_out_acc)
            if keep_pct != 1:
                left_out_accs.append(left_out_acc)

        if (args.save_model):
            torch.save(model.state_dict(), "mnist_cnn.pt")

        print(test_accs, left_out_accs, non_left_out_accs)
        plt.figure(1)
        plt.plot([x for x in range(1, len(test_accs) + 1)], test_accs, label="{0}%".format(100 * keep_pct))
        plt.figure(2)
        plt.plot([x for x in range(1, len(left_out_accs) + 1)], left_out_accs, label="{0}%".format(100 * keep_pct))
        plt.figure(3)
        plt.plot([x for x in range(1, len(non_left_out_accs) + 1)], non_left_out_accs,
                 label="{0}%".format(100 * keep_pct))

    plt.figure(1)
    plt.legend(keep_pcts, loc='upper left', title="Pct of digit/color combinations used in training")
    plt.xlim(1, 5)
    plt.ylim(0, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Overall Test Accuracy')
    plt.grid(True)
    plt.yticks([y / 10 for y in range(0, 11)])
    plt.xticks([x for x in range(1, 6)])
    plt.savefig("plots/" + str(args.jitter) + "_noisy_test_acc.pdf")

    plt.figure(2)
    plt.legend(keep_pcts[:-1], loc='upper left', title="Pct of digit/color combinations used in training")
    plt.xlim(1, 5)
    plt.ylim(0, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy on Left-Out Combinations')
    plt.grid(True)
    plt.yticks([y / 10 for y in range(0, 11)])
    plt.xticks([x for x in range(1, 6)])
    plt.savefig("plots/" + str(args.jitter) + "_noisy_left_out_acc.pdf")

    plt.figure(3)
    plt.legend(keep_pcts, loc='upper left', title="Pct of digit/color combinations used in training")
    plt.xlim(1, 5)
    plt.ylim(0, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy on Non-Left-Out Combinations')
    plt.grid(True)
    plt.yticks([y / 10 for y in range(0, 11)])
    plt.xticks([x for x in range(1, 6)])
    plt.savefig("plots/" + str(args.jitter) + "_noisy_non_left_out_acc.pdf")


if __name__ == '__main__':
    main()

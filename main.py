import argparse
from torch import cuda
import torch.nn as nn
import torch.optim as optim

import utils
import Models
import train_test


def Main():
    parser = argparse.ArgumentParser('DL_Module', add_help=False)
    # parser: 명령줄 인자를 parsing: 터미널 or cmd에서 사용하는 명령어를 프로그램 내에서 활용 가능한 형태로 변환
    # DL_Module: 프로그램명으로 도움말 옵션에 표시, add_help: --h, --help를 통한 도움말 옵션 비활성화

    parser.add_argument('--GPU', default=True, type=bool)
    parser.add_argument('--model', default='LeNet', type=str)
    parser.add_argument('--dataset', default="MNIST", type=str)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)

    args = parser.parse_args()  # 파싱

    print(args)

    device = 'cuda' if cuda.is_available() else 'cpu'

    print(f"dataset Load: {args.dataset}")

    if args.dataset == 'MNIST':
        train_loader, test_loader = utils.MNIST(args)

    print(f"{args.model} execution")

    if args.model == 'LeNet':

        model = Models.LeNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epoch+1):
        print(f"train[{epoch}] start")
        train_test.train(model, epoch, criterion,
                         optimizer, train_loader, device)
        print(f"test[{epoch}] start")
        train_test.test(model, epoch, criterion,
                        optimizer, test_loader, device)

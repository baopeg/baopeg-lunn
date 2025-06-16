import torch
import torchvision
from torch import optim,nn
from tqdm import tqdm
from LeNet5 import LeNet5
import argparse

def train(model,device,train_loader,optimizer,criterion):
    model.train()
    total_loss = 0
    for data,target in tqdm(train_loader):
        data,target = data.to(device),target.to(device)
        output= model(data)
        loss = criterion(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def test(model,device,test_loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data,target in tqdm(test_loader):
            data,target = data.to(device),target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    return correct / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--save_dir',type=str,default='model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size)


    model = LeNet5().to(device)
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    criterion = nn.CrossEntropyLoss()



    for epoch in range(1,args.epochs+1):
        train_loss = train(model,device,train_loader,optimizer,criterion)
        test_acc = test(model,device,test_loader)
        print(f'Epoch {epoch}: Loss={train_loss:.4f}, Accuracy={test_acc:.2%}')

    torch.save(model.state_dict(),f'{args.save_dir}/lenet.pth')


if __name__ == '__main__':
    main()


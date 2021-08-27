import pandas as pd
import numpy as np
import torch


def crossEntropyLossValue(tensor1,tensor2):
    '''
    you must rewrite your own crossEntropyLoss since
    the pytorch version of crossEntropyLoss is
    (p(x)*log(q(x))).sum()
    but the crossEntropyLoss applied in this paper is
    (p(x)*log(q(x))+(1-p(x))*log(1-q(x))).sum()
    '''
    # loss = (-tensor1*torch.log(tensor2)-(1-tensor1)*torch.log(1-tensor2)).sum()/tensor1.shape[0]
    loss = ((tensor1-tensor2)*(tensor1-tensor2)).sum()
    return loss

def read_csv():
    loss = 0
    data = pd.read_csv('fake_fingerprint.csv',header=None)
    mean = pd.read_csv('mean_targetData.csv',header=None)
    data_array = np.array(data)
    mean_array = np.array(mean)
    for i in range(7400):
        print(i)
        tensor1 = torch.from_numpy(data_array[i])
        tensor2 = torch.from_numpy(mean_array[0])
        loss = loss+crossEntropyLossValue(tensor1,tensor2)
    print(loss/7400)


if __name__ == '__main__':
    read_csv()
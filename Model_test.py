import math
import torch
from torch import nn
from torch.utils.data import DataLoader
import time

from MyDataset_a import PositionDataset
from PID_CNN import PID_CNN

# 模型实例化
model = PID_CNN()

# 载入预训练权重
model.load_state_dict(torch.load("pretrained_weights.pth", weights_only=True))

# Batch_size
Batch_size = 32
Train_shuffle = False

axis_list  = ['p1x','p1y','p1z','p2x','p2y','p2z','p3x','p3y','p3z','v1x','v1y','v1z','v2x','v2y','v2z','ax','ay','az']

def test_model(test_dataloader):
    # Loss
    criterion = nn.MSELoss()
    criterion2 = nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)
    criterion2.to(device)

    test_loss = 0.0
    test_num = 0
    test_num_axis = 0
    test_max_err = 0

    start_time = time.time()

    model.eval()
    with torch.no_grad():
        for batch, (index, inputs, pos_raw, pos_norm) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            pos_raw = pos_raw.to(device)
            pos_norm = pos_norm.to(device)
            # 预测结果
            pre_norm = model(inputs)
            pre_raw = pre_norm * std + mean
            ###############################################
            # 总损失
            loss2_ave = criterion2(pre_raw, pos_raw)
            loss2 = loss2_ave * pos_norm.numel()

            test_loss += loss2
            test_num += pos_norm.numel()

            test_num_axis += pos_norm[:, 0].numel()
            #  轴误差
            for axis in range(pos_raw.shape[1]):
                axis_pre = pre_raw[:, axis]
                axis_pos = pos_raw[:, axis]
                loss2_ave_axis = criterion2(axis_pre, axis_pos)
                loss2_axis = loss2_ave_axis * pos_norm[:, 0].numel()
                err_list1[axis][1] += loss2_axis.item()
                err_axis = axis_pre - axis_pos
                err_axis = abs(err_axis)
                err_values_axis, hang_index_axis = torch.max(err_axis, 0)
                max_index_axis = str(index[hang_index_axis].numpy() + 1)
                if err_values_axis > err_list1[axis][2]:
                    err_list1[axis][2] = err_values_axis.item()
                    err_list1[axis][3] = max_index_axis

        end_time = time.time()
        time_test =  end_time - start_time
        test_min = time_test // 60
        test_sec = time_test % 60

        print(f'test end, time: {test_min:.0f} min {test_sec:.2f} s')

        sum_p = 0
        for i in range(9):
            sum_p += err_list1[i][1]
        p_std = math.sqrt(sum_p/9/test_num_axis)

        sum_v = 0
        for i in range(9,15):
            sum_v += err_list1[i][1]
        v_std = math.sqrt(sum_v/6/test_num_axis)

        sum_a = 0
        for i in range(15,18):
            sum_a += err_list1[i][1]
        a_std = math.sqrt(sum_a / 3 / test_num_axis)

        print(f"p_std: {p_std}, v_std: {v_std}, a_std: {a_std}")

        print("------------------------------------------------------------------------------------------------")
        print("{:>10} {:>25} {:>25}".format('axis', "std error", "max error"))

        for i in range(len(err_list1)):
            err_list1[i][1] = err_list1[i][1] / test_num_axis
            err_list1[i][1] = math.sqrt(err_list1[i][1])
            print("{:>10} {:>25} {:>25}".format(axis_list[i],err_list1[i][1],err_list1[i][2]))

        print("------------------------------------------------------------------------------------------------")

    return 1


if __name__ == '__main__':

    # 理论均值方差
    mean = 0.0
    std = 28.86751345948128822

    test_data = PositionDataset(dataset_path=r"test_data_example")
    test_dataloader = DataLoader(test_data,batch_size=Batch_size,shuffle=Train_shuffle,num_workers=0)

    err_list1 = []
    test_input = test_data[0]
    for i in range(test_input[2].shape[0]):
        err_list1.append([i + 1])
        err_list1[i].append(0)
        err_list1[i].append(0)
        err_list1[i].append(0)

    test_model(test_dataloader)

pass

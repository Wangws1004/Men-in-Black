import os

import matplotlib.pyplot as plt
import torch


def test_step(
    batch, model, device, batch_idx, batch_size, show=False, file_output=False
):
    x, y, img_path = batch
    x = x.to(device)
    y = y.to(device)
    out = torch.sigmoid(model(x)["out"])
    confusion_mat = torch.zeros(
        (model.num_lanes, model.num_lanes), device=device, dtype=torch.long
    )
    f1_sum = 0
    f1_cnt = 0
    acc = torch.tensor(0.0, device=device)
    imshow = show
    if imshow:
        for i, output in enumerate(out):
            final_out = torch.argmax(output, 0)
            img = x[i].cpu().permute((1, 2, 0)).numpy()
            # img = img[:,:,::-1]
            plt.imsave("input.png", img)
            plt.imsave(
                "output.png", (final_out.cpu()).int(), vmin=0, vmax=model.num_lanes - 1
            )
            plt.imsave(
                "target.png", (y[i].cpu()).int(), vmin=0, vmax=model.num_lanes - 1
            )
            input()
    else:
        for i, output in enumerate(out):
            #             print(output.shape)
            final_out = torch.argmax(output, 0)
            #             print(final_out.shape)

            acc += torch.sum((final_out == y[i])) / (512 * 1024.0)

            for xx in torch.arange(model.num_lanes, device=device):
                for yy in torch.arange(model.num_lanes, device=device):
                    confusion_mat[xx, yy] += torch.sum((final_out == xx) * (y[i] == yy))

            aa, bb, cnt = 0, 0, 0
            for ii in range(model.num_lanes):
                if (
                    torch.sum(confusion_mat[ii, :]) != 0
                    and torch.sum(confusion_mat[:, ii]) != 0
                ):
                    aa += (
                        confusion_mat[ii, ii] / torch.sum(confusion_mat[ii, :]).float()
                    )
                    bb += (
                        confusion_mat[ii, ii] / torch.sum(confusion_mat[:, ii]).float()
                    )
                    cnt += 1
            aa /= cnt
            bb /= cnt
            # self.f1 += (2*aa*bb/(aa+bb))
            # self.f1cnt += 1
            f1 = (2 * aa * bb / (aa + bb)).item()
            f1_sum += f1
            f1_cnt += 1
            #             print(img_path[i], "F1 measure :", f1)

            #             file_output = False
            if file_output:
                if not os.path.exists("./outputs/"):
                    os.mkdir("./outputs/")
                img = x[i].cpu().permute((1, 2, 0)).numpy()
                folder_path = "./outputs/" + str(batch_idx * batch_size + i)
                #                 print(folder_path)
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                plt.imsave(folder_path + "/input.png", img)
                plt.imsave(
                    folder_path + "/output.png",
                    (final_out.cpu()).int(),
                    vmin=0,
                    vmax=model.num_lanes - 1,
                )
                plt.imsave(
                    folder_path + "/target.png",
                    (y[i].cpu()).int(),
                    vmin=0,
                    vmax=model.num_lanes - 1,
                )
        acc /= len(out)

        return confusion_mat.cpu().numpy(), f1_sum, f1_cnt, img_path


def test_epoch_end(outputs):
    sum_confusion_mat = 0
    total_f1 = 0
    total_f1_cnt = 0
    for confusion_mat, f1_sum, f1_cnt, _ in outputs:
        sum_confusion_mat += confusion_mat
        total_f1 += f1_sum
        total_f1_cnt += f1_cnt

    #     print("total_f1_cnt",total_f1_cnt)
    #     print("average F1 measure", total_f1/total_f1_cnt)
    #     print("total confusion matrix:\n", sum_confusion_mat.cpu().numpy())
    return total_f1 / total_f1_cnt, sum_confusion_mat

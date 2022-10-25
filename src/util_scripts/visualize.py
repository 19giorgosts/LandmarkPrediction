import numpy as np
import pandas as pd

def save_matches(outputs,batch_idx,output_path):
    keypoints_detected = outputs['matches'][0, :, :]
    print(np.nonzero(keypoints_detected)[0:5])
    indices=np.nonzero(keypoints_detected)

    pd.DataFrame(outputs['landmarks_1'][0, :, :].cpu().detach().numpy()).to_csv("{}/img_{}_landmarks1.csv".format(output_path,batch_idx), header=None,  index=None)
    pd.DataFrame(outputs['landmarks_2'][0, :, :].cpu().detach().numpy()).to_csv("{}/img_{}_landmarks2.csv".format(output_path,batch_idx), header=None,  index=None)
    pd.DataFrame(outputs['kpt_sampling_grid_1'][0, 0, 0, :, :].cpu().detach().numpy()).to_csv("{}/img_{}_kpt_sampling_grid_1.csv".format(output_path, batch_idx), header=None, index=None)
    pd.DataFrame(outputs['kpt_sampling_grid_2'][0, 0, 0, :, :].cpu().detach().numpy()).to_csv("{}/img_{}_kpt_sampling_grid_2.csv".format(output_path, batch_idx), header=None, index=None)
    pd.DataFrame(outputs['matches'][0, :, :].cpu().detach().numpy()).to_csv("{}/img_{}_matches.csv".format(output_path,batch_idx), index=None)

    kpts1=[]
    kpts2=[]
    en=0
    for j in indices:
        # print(type(outputs['landmarks_1'][0, j[0].item(), :].tolist()))
        kpts1.append(outputs['landmarks_1'][0, j[0].item(), :].tolist())
        kpts2.append(outputs['landmarks_2'][0, j[1].item(), :].tolist())
        if en<5:
            print(outputs['landmarks_1'][0, j[0].item(), :].tolist())
            print(outputs['landmarks_2'][0, j[1].item(), :].tolist())
        en+=1
    # print(outputs['landmarks_1'][0, :, :])

    with open('{}/img_{}matches1.txt'.format(output_path,batch_idx), 'w') as f:
        for line in kpts1:
            f.write(f"{line}\n")

    with open('{}/img_{}matches2.txt'.format(output_path,batch_idx), 'w') as f:
        for line in kpts2:
            f.write(f"{line}\n")
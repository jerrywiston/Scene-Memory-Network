import numpy as np
import os

def read_dataset(path):
    flist = os.listdir(path)
    dataset_color = None
    dataset_depth = None
    dataset_pose = None
    for fname in flist:
        data_path = os.path.join(path, fname)
        data = np.load(data_path, allow_pickle=True)
        print("Load", data_path)
        if dataset_color is None:
            dataset_color = data["color"].astype(np.float32) / 255
            dataset_depth = data["depth"]
            dataset_pose = data["pose"]
        else:
            dataset_color = np.concatenate([dataset_color, data["color"].astype(np.float32) / 255], 0)
            dataset_depth = np.concatenate([dataset_depth, data["depth"]], 0)
            dataset_pose = np.concatenate([dataset_pose, data["pose"]], 0)
        
    print(dataset_color.shape)
    print(dataset_depth.shape)
    print(dataset_pose.shape)
    return dataset_color, dataset_depth, dataset_pose

def read_dataset_color(path):
    flist = os.listdir(path)
    dataset_color = None
    dataset_pose = None
    for fname in flist:
        data_path = os.path.join(path, fname)
        data = np.load(data_path, allow_pickle=True)
        print("Load", data_path)
        if dataset_color is None:
            dataset_color = data["color"].astype(np.float32) / 255
            dataset_pose = data["pose"]
        else:
            dataset_color = np.concatenate([dataset_color, data["color"].astype(np.float32) / 255], 0)
            dataset_pose = np.concatenate([dataset_pose, data["pose"]], 0)
        
    print(dataset_color.shape)
    print(dataset_pose.shape)
    return dataset_color, dataset_pose

def get_batch(color, pose, obs_size=12, batch_size=32, to_torch=True):
    img_obs = None
    pose_obs = None
    img_query = None
    pose_query = None
    for i in range(batch_size):
        batch_id = np.random.randint(0, color.shape[0])
        obs_id = np.random.randint(0, color.shape[1], size=obs_size)
        query_id = np.random.randint(0, color.shape[1])
        
        if img_obs is None:
            img_obs = color[batch_id:batch_id+1, obs_id].reshape(-1,color.shape[-3], color.shape[-2], color.shape[-1])
            pose_obs = pose[batch_id:batch_id+1, obs_id].reshape(-1,pose.shape[-1])
            img_query = color[batch_id:batch_id+1, query_id]
            pose_query = pose[batch_id:batch_id+1, query_id]
        else:
            img_obs = np.concatenate([img_obs, color[batch_id:batch_id+1, obs_id].reshape(-1,color.shape[-3], color.shape[-2], color.shape[-1])], 0)
            pose_obs = np.concatenate([pose_obs, pose[batch_id:batch_id+1, obs_id].reshape(-1,pose.shape[-1])], 0)
            img_query = np.concatenate([img_query, color[batch_id:batch_id+1, query_id]], 0)
            pose_query = np.concatenate([pose_query, pose[batch_id:batch_id+1, query_id]], 0)
    
    if to_torch:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_obs = torch.FloatTensor(img_obs).permute(0,3,1,2).to(device)
        pose_obs = torch.FloatTensor(pose_obs).to(device)
        img_query = torch.FloatTensor(img_query).permute(0,3,1,2).to(device)
        pose_query = torch.FloatTensor(pose_query).to(device)

    return img_obs, pose_obs, img_query, pose_query

def get_file_batch(path, obs_size=12, batch_size=32, to_torch=True):
    flist = os.listdir(path)
    file_id = np.random.randint(0, len(flist))
    data_path = os.path.join(path, flist[file_id])
    data = np.load(data_path, allow_pickle=True)
    color = data["color"].astype(np.float32) / 255
    pose = data["pose"]

    img_obs = None
    pose_obs = None
    img_query = None
    pose_query = None
    for i in range(batch_size):
        batch_id = np.random.randint(0, color.shape[0])
        obs_id = np.random.randint(0, color.shape[1], size=obs_size)
        query_id = np.random.randint(0, color.shape[1])
        
        if img_obs is None:
            img_obs = color[batch_id:batch_id+1, obs_id].reshape(-1,color.shape[-3], color.shape[-2], color.shape[-1])
            pose_obs = pose[batch_id:batch_id+1, obs_id].reshape(-1,pose.shape[-1])
            img_query = color[batch_id:batch_id+1, query_id]
            pose_query = pose[batch_id:batch_id+1, query_id]
        else:
            img_obs = np.concatenate([img_obs, color[batch_id:batch_id+1, obs_id].reshape(-1,color.shape[-3], color.shape[-2], color.shape[-1])], 0)
            pose_obs = np.concatenate([pose_obs, pose[batch_id:batch_id+1, obs_id].reshape(-1,pose.shape[-1])], 0)
            img_query = np.concatenate([img_query, color[batch_id:batch_id+1, query_id]], 0)
            pose_query = np.concatenate([pose_query, pose[batch_id:batch_id+1, query_id]], 0)
    
    if to_torch:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_obs = torch.FloatTensor(img_obs).permute(0,3,1,2).to(device)
        pose_obs = torch.FloatTensor(pose_obs).to(device)
        img_query = torch.FloatTensor(img_query).permute(0,3,1,2).to(device)
        pose_query = torch.FloatTensor(pose_query).to(device)

    return img_obs, pose_obs, img_query, pose_query


if __name__ == "__main__":
    color_data, pose_data = read_dataset_color("Datasets/MazeBoardRandom/")
    img_obs, pose_obs, img_query, pose_query = get_batch(color_data, pose_data, to_torch=False)
    print(img_obs.shape, pose_obs.shape, img_query.shape, pose_query.shape)

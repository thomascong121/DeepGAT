from tqdm.auto import tqdm
import os
import pandas as pd


root = '/home/congz3414050/HistoGCN/data/5X/Tumor'
out = '/home/congz3414050/HistoGCN/data/5X/csvs'

def path2csv(root, out):
    count = 0
    for tumors in tqdm(os.listdir(root)):
        content = {'node':[], 'coord':[], 'label':[], 'id':[]}
        tumor_pth = os.path.join(root, tumors)
        ids = 0

        for sub_root in os.listdir(tumor_pth):
            sub_root_pth = os.path.join(tumor_pth, sub_root, 'image')
            if len(os.listdir(sub_root_pth)) == 0:
                break
            for node in os.listdir(sub_root_pth):
                node_pth = os.path.join(sub_root_pth, node)
                node_coor_combine = node.split('.png')[0]
                node_label = sub_root
                content['node'].append(node_pth)
                content['coord'].append(node_coor_combine)
                content['label'].append(node_label)
                content['id'].append(ids)
                ids += 1
            count += 1
            #     print(content)
            #     break
            # break
        if len(content['node'])!=0:
            out_csv_pth = os.path.join(out, '%s.csv'%tumors)
            df = pd.DataFrame.from_dict(content)
            df.to_csv(out_csv_pth, index=False)

    print('total %d csvs'%count)

def obtain_class(root):
    content = {'image':[],'label':[]}
    # prefix = 'F:\BaiduNetdiskDownload\CAMELYON16\GCN\data'
    for tumor_csv in os.listdir(root):
        csv_path = os.path.join(root, tumor_csv)
        df = pd.read_csv(csv_path)
        print(df.head())
        node_list = []
        for node in df['node'].tolist():
            # node_sufux= node.split(prefix)[1]
            # node_path = prefix + r'\5X' + node_sufux
            node_list.append(node)

        content['image'] += node_list
        content['label'] += df['label'].tolist()

    out_csv_pth = os.path.join(root, 'all_data.csv')
    df = pd.DataFrame.from_dict(content)
    df.to_csv(out_csv_pth, index=False)
    print(out_csv_pth)

# path2csv(root, out)
obtain_class(out)
# df = pd.read_csv(r'F:\BaiduNetdiskDownload\CAMELYON16\GCN\data\5X\csvs\all_data.csv')
# df_normal = df[df['label']=='Normal']
# df_tumor = df[df['label']=='Tumor']
# print(len(df_normal),len(df_tumor))
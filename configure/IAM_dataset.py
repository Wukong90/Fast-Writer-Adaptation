import numpy as np
from skimage import io as img_io
from skimage import transform
from skimage import util
from tqdm import tqdm

# PATH TO IAM DATASET ON SSD
root_path = './data/'
line_gt = root_path + 'IAM/lines.txt'
line_img = root_path + 'IAM/lines/'
line_train_wid = root_path + 'IAM/split/trainset_shuf_trainid.txt' #For train wid
line_train_rec = root_path + 'IAM/split/trainset.txt'

line_test = root_path + 'IAM/split/testset.txt'

line_val1_wid = root_path + 'IAM/split/trainset_shuf_cvid.txt' #For train wid
line_val1_rec = root_path + 'IAM/split/validationset1.txt'


line_val2 = root_path + 'IAM/split/validationset2.txt'

#For idloss
img2id_f = open('./data/IAM/split/train_img2id.txt','r')
img2id = {}
ids = {}
count=0
while True:
	line=img2id_f.readline()
	line=line.strip("\n")
	if not line:
		break
	tmp_line=line.split("	")
	if tmp_line[1] not in ids:
		ids[tmp_line[1]] = count
		count = count + 1
	img2id[tmp_line[0]] = ids[tmp_line[1]]
# ------------------------------------------------

def gather_iam_line(set='train',set_wid=False):

    gtfile = line_gt
    root_path = line_img
    if set == 'train' and set_wid == True:
        data_set = np.loadtxt(line_train_wid, dtype=str)
    elif set == 'train' and set_wid == False:
        data_set = np.loadtxt(line_train_rec, dtype=str)

    elif set == 'test':
        data_set = np.loadtxt(line_test, dtype=str)

    elif set == 'val' and set_wid == True:
        data_set = np.loadtxt(line_val1_wid, dtype=str)
    elif set == 'val' and set_wid == False:
        data_set = np.loadtxt(line_val1_rec, dtype=str)

    elif set == 'val2':
        data_set = np.loadtxt(line_val2, dtype=str)
    else:
        print("Cannot find this dataset. Valid values for set are 'train', 'test', 'val' or 'val2'.")
        return
    gt = []
    print("Reading IAM dataset...")
    for line in open(gtfile):
        if not line.startswith("#"):
            info = line.strip().split()
            name = info[0]
            name_parts = name.split('-')
            pathlist = [root_path] + ['-'.join(name_parts[:i+1]) for i in range(len(name_parts))]
            line_name = pathlist[-1]
            if (info[1] != 'ok') or (line_name not in data_set):  # if the line is not properly segmented, i.e., only "proper" sentences were used in this paper.
                continue
            #if (line_name not in data_set):
	    #	continue
            if (set_wid == True):   
                wid = img2id[line_name]
            else:
                if (set == 'train'):
                    wid=img2id[line_name]
                else:
                    wid=0

            img_path = '/'.join(pathlist)
            transcr = ' '.join(info[8:])
            gt.append((img_path, transcr, wid))
    print("Reading done.")
    return gt


def iam_main_loader(set='train',set_wid=False):

    line_map = gather_iam_line(set,set_wid)

    data = []
    for i, (img_path, transcr, wid) in enumerate(tqdm(line_map)):
        try:
            img = img_io.imread(img_path + '.png')
            img = 1 - img.astype(np.float32) / 255.0

        except:
            continue
        data += [(img, transcr.replace("|", " "), wid)]


    return data

if __name__ == '__main__':
    (img_path, transcr) = gather_iam_line('train')[0]
    img = img_io.imread(img_path + '.png')

    data = iam_main_loader(set='train')
    print("length of train set:", len(data))

    data = iam_main_loader(set='test')
    print("length of test set:", len(data))

    data = iam_main_loader(set='val')
    print("length of val set:", len(data))
    print("Success")

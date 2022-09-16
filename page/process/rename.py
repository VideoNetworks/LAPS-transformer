import os

root = 'train'

folders = os.listdir(root)

for one in folders:
    print(one)
    src = os.path.join(root, one)
    new = one.strip().replace(' ', '_')
    new = new.strip().replace('"', '')
    print(new)
    dst = os.path.join(root, new)
    os.rename(src, dst)

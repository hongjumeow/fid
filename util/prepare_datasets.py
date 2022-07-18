import os

def is_image_file(name):
    if name.endswith('.jpg' or '.png'):
        return True
    else:
        return False

def walk_dirs_and_append(dir):
    images = []
    for root, dirs, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

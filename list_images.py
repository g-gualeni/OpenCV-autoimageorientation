import glob


def files(path, extension):
    path2 = path.strip()
    path2 = path2.replace('\\', '/')
    path2 = path2.rstrip('/') + '/'
    glob_arg = path2 + "*." + extension.strip()
    print(glob_arg)
    file_list = glob.glob(glob_arg)
    return file_list




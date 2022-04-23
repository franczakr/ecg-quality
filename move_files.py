import os
from os import listdir
from os.path import isfile, join


def move_files(images_path: str, csv_path: str, csv_new_path: str):
    files_to_move = [f
                     for f in listdir(images_path)
                     if isfile(join(images_path, f))]

    if not os.path.exists(csv_new_path):
        os.makedirs(csv_new_path)

    for file in files_to_move:
        filename = file[:-4] + '.csv'
        old_path = csv_path + '/' + filename
        if isfile(old_path):
            new_path = csv_new_path + '/' + filename
            os.rename(old_path, new_path)


def main():
    move_files('data/slices_af/watpliwe', 'data/slices_af/csv', 'data/slices_af/csv/watpliwe')
    move_files('data/slices_bq/watpliwe', 'data/slices_bq/csv', 'data/slices_bq/csv/watpliwe')
    move_files('data/slices_gq/watpliwe', 'data/slices_gq/csv', 'data/slices_gq/csv/watpliwe')
    move_files('data/slices_gq/zle', 'data/slices_gq/csv', 'data/slices_gq/csv/zle')
    move_files('data/slices_others/watpliwe', 'data/slices_otheres/csv', 'data/slices_others/csv/watpliwe')


if __name__ == '__main__':
    main()

import glob
import os


def two_values_before_number(filebase, file_format='png'):
    for index, filename in enumerate(glob.glob(filebase)):
        vals = filename.split('-')
        new_filename = ('test@' + vals[0] + '-' + vals[1] + '-' +
                        str(index) + '.' + file_format)
        os.rename(filename, new_filename)
    for filename in (glob.glob('test@*')):
        vals = filename.split('@')
        os.rename(filename, vals[1])


def one_value_before_number(filebase, file_format='png'):
    for index, filename in enumerate(glob.glob(filebase)):
        vals = filename.split('-')
        new_filename = 'test@' + vals[0] + '-' + str(index) + '.' + file_format
        os.rename(filename, new_filename)
    for filename in (glob.glob('test@*')):
        vals = filename.split('@')
        os.rename(filename, vals[1])


def main():
    two_values_before_number('annotated*')
    one_value_before_number('signs*')
    one_value_before_number('bboxes*', file_format='json')
    two_values_before_number('carla-center*')
    two_values_before_number('carla-left*')
    two_values_before_number('carla-right*')
    two_values_before_number('carla-lidar*', file_format='ply')
    two_values_before_number('carla-segmented*')
    two_values_before_number('perfect-detector*')
    two_values_before_number('carla-depth*', file_format='pkl')

if __name__ == '__main__':
    main()

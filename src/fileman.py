import os
import time
from datetime import datetime
from shutil import copyfile

from pathlib import Path
'''
from filter_redmine_csv import filter_csv
from outmail import send_mail'''


def get_datetime(rev: bool = False):
    if rev:
        date_time = str(datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    else:
        date_time = str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
    return date_time


class ListDir:
    def __init__(self, target_path):
        """

        :param target_path: Path for detect difference
        """
        self.target_path = Path(target_path)
        self.reference_filelist = self.get_listdir()

    def get_listdir(self):
        return set(os.listdir(self.target_path))

    def get_diff(self):
        return self.get_listdir() - self.reference_filelist


def clear_testpath(testpath):
    if 'test' in str(testpath):
        if len(os.listdir(testpath)) > 0:
            for file in os.listdir(testpath):
                os.remove(testpath / Path(file))
        os.rmdir(testpath)
        if not Path(testpath).is_dir():
            print(f'testdir {testpath} removed')


def wait_downloaded_file(driver, pathtosave):
    my_list = ListDir(pathtosave)
    downloaded_file_name = set()
    counter = 15
    downloaded_file_full_path = None
    while not downloaded_file_name and counter:
        time.sleep(1)
        downloaded_file_name = my_list.get_diff()
        if len(downloaded_file_name) == 1 \
                and 'issues' in str(downloaded_file_name) \
                and '.csv' in str(downloaded_file_name) \
                and 'down' not in str(downloaded_file_name):
            driver.close()
            driver.quit()
            downloaded_file_full_path = Path(pathtosave) / Path(list(downloaded_file_name)[0])
        else:
            downloaded_file_name = set()
        counter -= 1
        print(f'осталось {counter} секунд')
    if downloaded_file_full_path \
            and downloaded_file_full_path.exists():
        print(f'сохранен {downloaded_file_full_path}')
        clear_testpath(pathtosave)
    else:
        print('Downloaded file not exist. ERROR')
    return downloaded_file_full_path


def wait_new_file(func):
    def wrapper(pathtosave):
        my_list = ListDir(pathtosave)
        print('start')
        func(pathtosave)
        print('finish')
        return my_list.get_diff()

    return wrapper


'''def send_error_to_me(error_message):
    send_mail(e_addr='phaggi@gmail.com', subject='Problem!',
              body_text=error_message)
'''

'''def mailme_error(error_message):
    send_error_to_me(error_message=error_message)
    raise Exception(error_message)'''


'''def find_file(filenames: set, pathtosave, word):
    for filename in filenames:
        if word in filename:
            return pathtosave / Path(filename)
        else:
            mailme_error(f'В папке {pathtosave} новых файлов не найдено. А должны быть!')'''


'''def prepare_vba_school_input_csv(downloaded_file_full_path, vba_input_file_name, vba_work_dir):
    src = downloaded_file_full_path
    dst = Path.cwd() / Path(vba_work_dir) / Path(vba_input_file_name)
    try:
        copyfile(src, dst)
        tmpfile = filter_csv(src)
        copyfile(tmpfile, dst)
        return dst
    except FileNotFoundError as e:
        mailme_error(f'Не удалось подготовить входной файл для VBA обработки.\nОшибка {e}')'''


def prepare_filename(result_filename=None, ext=None):
    """

    :param result_filename: str lines for filename
    :param ext: str suffix '.xxx' or 'xxx'
    :return: (Path(filename.ext|, filename, ext)
    """
    if not result_filename:
        result_filename = 'выгрузка'
    if not ext:
        ext = 'xlsx'
    elif ext.startswith('.'):
        ext = ext[1:]
    result_filename = '_'.join([result_filename, get_datetime()])
    return Path('.'.join([result_filename, ext])), result_filename, ext


def move_result(result_filename=None, ext=None, src=None, dst=None):
    """

    :param result_filename: str lines for filename
    :param ext: str suffix '.xxx' or 'xxx'
    :param src: maybe Path(/dir/file) or default vba/result.xlsx
    :param dst: maybe Path(/dir/file) or default ~/redmine/(result_filename)
    :return: (dst, result_filename)
    """
    result_full_filename = prepare_filename(result_filename=result_filename, ext=ext)[0]
    if not src:
        src = Path.cwd() / Path('vba') / Path('result.xlsx')
    if not dst:
        dst = Path.home() / Path('redmine') / result_full_filename
    copyfile(src, dst)
    os.remove(src)
    return dst, result_full_filename


if __name__ == '__main__':
    pass

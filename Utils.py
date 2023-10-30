import logging
import os
import sys

def rating_to_label_gains(rating):
    rating2label_dict = {'Perfect' : 31,
                         'Excellent' : 15,
                         'Good' : 7,
                         'Fair' : 3,
                         'Bad' : 0,
                         'CantJudgeForeign' : 0,
                         'CantJudgePDNL' : 0,
                         'CantJudgeLogin' : 0,
                         'CantJudgeAdult' : 0,
                         'Det' : 0}
    return rating2label_dict[rating]

def rating_to_label(rating):
    rating2label_dict = {'Perfect' : 5,
                         'Excellent' : 4,
                         'Good' : 3,
                         'Fair' : 2,
                         'Bad' : 0,
                         'CantJudgeForeign' : 0,
                         'CantJudgePDNL' : 0,
                         'CantJudgeLogin' : 0,
                         'CantJudgeAdult' : 0,
                         'Det' : 0}

    if rating in rating2label_dict:
        return rating2label_dict[rating]
    else:
        return rating

def init_logger(output_dir: str, output_file: str ="training.log") -> logging.Logger:
    log_format = "[%(levelname)s] %(asctime)s - %(message)s"
    log_dateformat = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(format=log_format, datefmt=log_dateformat, stream=sys.stdout, level=logging.INFO)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(os.path.join(output_dir, output_file))
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(__name__)


import requests, urllib3

def send_to_bark(title, content):
       bl = "https://api.day.app/FDBAQFgsX7rAt2GSBk8Tm6"
       urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
       try:
           msg = "{0}/{1}/{2}/?isArchive=1".format(bl, title, content)
           link = msg
           res = requests.get(link, verify=False)
       except Exception as e:
           print('Reason:', e)
           return
       return

if __name__ == '__main__':
    send_to_bark("Test", "test")

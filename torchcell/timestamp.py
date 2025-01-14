import datetime


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

from config.hparams import Hparams
from utils.log import Log
import train

def main(hparams):
    log = Log(hparams)
    hparams.logger = log.logger
    train.train(hparams)


if __name__ == "__main__":
    hparams = Hparams()
    main(hparams)


import argparse
import logging

import api
import framework

_CHOICES = {
    'api': api,
    'framework': framework,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=_CHOICES.get, choices=_CHOICES.values())
    parser.add_argument('-v', '--verbose', action='count', default=0)
    namespace, args = parser.parse_known_args()

    logging.basicConfig(level=logging.WARNING - namespace.verbose * 10)
    return namespace.mode.main(args)


if __name__ == '__main__':
    main()

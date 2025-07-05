import argparse
import importlib
import logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--open', action='store_const', default='closed',
                        const='open')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    namespace, args = parser.parse_known_args()

    logging.basicConfig(level=logging.WARNING - namespace.verbose * 10)
    return importlib.import_module(namespace.open).main(args)


if __name__ == '__main__':
    main()

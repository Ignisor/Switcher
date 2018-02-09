import argparse

from model import Switcher


parser = argparse.ArgumentParser()
parser.add_argument('text', nargs='?', type=str, help=f'Text to predict on')


if __name__ == '__main__':
    args = parser.parse_args()

    m = Switcher()
    m.init_model()

    result = m.predict(args.text)

    print('\n')
    print(f'Probability that operator is needed - {int(result[0][0] * 100)}%')

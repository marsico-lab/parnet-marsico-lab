import click

from .bin import predict, train


@click.group()
def main():
    pass


main.add_command(train.main, name='train')
main.add_command(predict.main, name='predict')

if __name__ == '__main__':
    main()

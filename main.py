import os
import sys
import click
from pathlib import Path

lib_path = Path(__file__).resolve().parents[0]
sys.path.append(Path(lib_path).__str__())
if os.environ.__contains__("PYTHONPATH") and len(os.environ["PYTHONPATH"]) > 0:
    delim = ';' if 'win' in sys.platform else ':'
    os.environ['PYTHONPATH'] += delim + lib_path.__str__()
else:
    os.environ["PYTHONPATH"] = lib_path.__str__()

from trader.train.supervised.gen_model import gen_model
from common.modules.dotdict import Dotdict
# from trader.common.modules.Logger import Logger
# Logger.init_log(os.path.join(lib_path, 'log_{}.log'.format(dt.datetime.today().date())))


@click.group()
@click.pass_context
def main(ctx: object) -> None:
    """
        Entry point for strategy generator and research environment
        :param ctx: Context from Click.
        :return: None
        """
    ctx.obj = Dotdict(dict(
        fn_params='ethusd_train',
        fn_settings='settings'
    ))


main.add_command(gen_model)

if __name__ == '__main__':
    main()

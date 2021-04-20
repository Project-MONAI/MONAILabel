import argparse
import json
import logging
import os
import sys

from monailabel.core.config import settings
from monailabel.interface import MONAILabelApp
from monailabel.interface.exception import MONAILabelError, MONAILabelException
from monailabel.utils.class_utils import get_class_of_subclass_from_file

logger = logging.getLogger(__name__)
app = None


def get_app_instance():
    return app_instance(settings.APP_DIR, settings.STUDIES)


def app_instance(app_dir, studies):
    global app
    if app is not None:
        return app

    logger.info(f"Initializing App from: {app_dir}; studies: {studies}")

    main_py = os.path.join(app_dir, 'main.py')
    if not os.path.exists(main_py):
        raise MONAILabelException(MONAILabelError.APP_INIT_ERROR, f"App Does NOT have main.py")

    c = get_class_of_subclass_from_file("main", main_py, MONAILabelApp)
    if c is None:
        raise MONAILabelException(MONAILabelError.APP_INIT_ERROR, f"App Does NOT Implement MONAILabelApp in main.py")

    o = c(app_dir=app_dir, studies=studies)
    methods = ["infer", "train", "info", "next_sample", "save_label"]
    for m in methods:
        if not hasattr(o, m):
            raise MONAILabelException(MONAILabelError.APP_INIT_ERROR, f"App Does NOT Implement '{m}' method in main.py")

    app = o
    return app


def save_result(result, output):
    print(json.dumps(result))
    if output:
        with os.open(output, "w") as fp:
            json.dumps(result, fp, indent=2)


def test_inference(args):
    app = app_instance(app_dir=args.app, studies=args.studies)
    request = json.loads(args.request)

    res_img, res_json = app.infer(request=request)
    result = {
        "label": res_img,
        "params": res_json
    }
    save_result(result, args.output)


def test_train(args):
    app = app_instance(app_dir=args.app, studies=args.studies)
    request = json.loads(args.request)
    result = app.train(request)
    save_result(result, args.output)


def test_info(args):
    app = app_instance(app_dir=args.app, studies=args.studies)
    result = app.info()
    save_result(result, args.output)


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--app', required=True)
    parser.add_argument('-s', '--studies', required=True)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('-l', '--logprefix', type=str, default='')
    parser.add_argument('-d', '--debug', action='store_true')

    subparsers = parser.add_subparsers(help='sub-command help')

    parser_a = subparsers.add_parser('infer', help='infer help')
    parser_a.add_argument('-r', '--request', required=True)
    parser_a.set_defaults(test='infer')

    parser_b = subparsers.add_parser('train', help='train help')
    parser_b.add_argument('-r', '--request', required=True)
    parser_b.set_defaults(test='train')

    parser_c = subparsers.add_parser('info', help='info help')
    parser_c.set_defaults(test='info')

    args = parser.parse_args()
    if not hasattr(args, 'test'):
        parser.print_usage()
        exit(-1)

    for arg in vars(args):
        print('USING:: {} = {}'.format(arg, getattr(args, arg)))
    print("")

    sys.path.append(args.app)
    sys.path.append(os.path.join(args.app, 'lib'))

    logging.basicConfig(
        level=(logging.DEBUG if args.debug else logging.INFO),
        format='[%(asctime)s] [%(levelname)s] (%(name)s) - %(message)s')

    if args.test == 'infer':
        test_inference(args)
    elif args.test == 'train':
        test_train(args)
    elif args.test == 'info':
        test_info(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    run_main()

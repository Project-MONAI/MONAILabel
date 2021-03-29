import argparse
import distutils.util
import json
import logging
import os
import pathlib
import shutil

from main import DeepgrowApp

logger = logging.getLogger(__name__)


def test_inference(args):
    output_path = os.path.dirname(args.output)
    os.makedirs(output_path, exist_ok=True)

    app = DeepgrowApp(name=args.name, app_dir=args.app_dir)
    logger.info('Running Inference: {}'.format(args.name))

    res_img = None
    res_json = None
    for it in range(args.runs):
        request = {
            "image": args.input,
            "params": json.loads(args.params),
            "device": args.device
        }
        res_img, res_json = app.infer(request=request)

    if res_img:
        file_ext = ''.join(pathlib.Path(res_img).suffixes)
        result_image = os.path.join(args.output, 'label_' + args.name + file_ext)

        shutil.move(res_img, result_image)
        os.chmod(result_image, 0o777)
        print('Check Result file: {}'.format(result_image))

    print('Result JSON: {}'.format(res_json))


def test_train(args):
    logger.info("Not Implemented Yet...")
    pass


def strtobool(val):
    return bool(distutils.util.strtobool(val))


def run_main():
    app_dir = os.path.realpath(os.path.join(os.path.dirname(__file__)))
    dataset_dir = os.path.realpath(os.path.join(app_dir, '..', '..', 'datasets', 'Task09_Spleen'))
    params = {
        "foreground": [[73, 177, 105], [66, 160, 105], [61, 200, 105]],
        "background": []
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='deepgrow_2d')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-a', '--app_dir', default='.')
    parser.add_argument('--device', default='cuda')

    subparsers = parser.add_subparsers(help='sub-command help')

    parser_a = subparsers.add_parser('infer', help='infer help')
    parser_a.add_argument('-i', '--input', default='/workspace/Data/_image.nii.gz')
    parser_a.add_argument('-o', '--output', default='/workspace/Data/')
    parser_a.add_argument('-p', '--params', default=json.dumps(params))
    parser_a.add_argument('-r', '--runs', type=int, default=1)
    parser_a.set_defaults(test='infer')

    parser_b = subparsers.add_parser('train', help='train help')
    parser_b.add_argument('-i', '--input', default=f"{dataset_dir}/dataset.json")
    parser_b.add_argument('-o', '--output', default=f"{app_dir}/model/run_0")
    parser_b.add_argument('-r', '--dataset_root', default=dataset_dir)
    parser_b.add_argument('-e', '--epochs', type=int, default=1)
    parser_b.add_argument('--amp', type=strtobool, default='true')
    parser_b.set_defaults(test='train')

    args = parser.parse_args()
    if not hasattr(args, 'test'):
        parser.print_usage()
        exit(-1)

    logging.basicConfig(
        level=(logging.DEBUG if args.debug else logging.INFO),
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    for arg in vars(args):
        logger.info('USING:: {} = {}'.format(arg, getattr(args, arg)))
    print("")

    if args.test == 'infer':
        test_inference(args)
    elif args.test == 'train':
        test_train(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    run_main()

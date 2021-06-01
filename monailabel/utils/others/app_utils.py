import argparse
import json
import logging
import os
import sys

from monailabel.interfaces import MONAILabelApp, MONAILabelError, MONAILabelException
from monailabel.utils.others.class_utils import get_class_of_subclass_from_file

logger = logging.getLogger(__name__)
app = None


def app_instance(app_dir, studies):
    global app
    if app is not None:
        return app

    logger.info(f"Initializing App from: {app_dir}; studies: {studies}")

    main_py = os.path.join(app_dir, "main.py")
    if not os.path.exists(main_py):
        raise MONAILabelException(MONAILabelError.APP_INIT_ERROR, "App Does NOT have main.py")

    c = get_class_of_subclass_from_file("main", main_py, MONAILabelApp)
    if c is None:
        raise MONAILabelException(
            MONAILabelError.APP_INIT_ERROR,
            "App Does NOT Implement MONAILabelApp in main.py",
        )

    o = c(app_dir=app_dir, studies=studies)
    app = o
    return app


def save_result(result, output):
    print(json.dumps(result))
    if output:
        with open(output, "w") as fp:
            json.dump(result, fp, indent=2)


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--app", required=True)
    parser.add_argument("-s", "--studies", required=True)
    parser.add_argument("-m", "--method", required=True)
    parser.add_argument("-r", "--request", type=str, default="{}")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-d", "--debug", action="store_true")

    args = parser.parse_args()
    for arg in vars(args):
        print("USING:: {} = {}".format(arg, getattr(args, arg)))
    print("")

    sys.path.append(args.app)
    sys.path.append(os.path.join(args.app, "lib"))

    logging.basicConfig(
        level=(logging.DEBUG if args.debug else logging.INFO),
        format="[%(asctime)s] [%(levelname)s] (%(name)s) - %(message)s",
    )

    a = app_instance(app_dir=args.app, studies=args.studies)
    request = json.loads(args.request)
    result = None

    if args.method == "infer":
        res_img, res_json = a.infer(request=request)
        result = {"label": res_img, "params": res_json}
    elif args.method == "train":
        result = a.train(request)
    elif args.method == "info":
        result = a.info(request)
    elif args.method == "batch_infer":
        result = a.batch_infer(request)
    elif args.method == "scoring":
        result = a.scoring(request)
    else:
        parser.print_help()
        exit(-1)

    save_result(result, args.output)


if __name__ == "__main__":
    run_main()

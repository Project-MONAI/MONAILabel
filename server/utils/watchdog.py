import logging

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

event_handler = None

logger = logging.getLogger(__name__)


class WatchDogHandler(PatternMatchingEventHandler):

    def __init__(self, patterns, context):
        super(WatchDogHandler, self).__init__(patterns=patterns)
        self.context = context

    def on_modified(self, event):
        super(WatchDogHandler, self).on_modified(event)

        logger.debug('File {} was just modified'.format(event.src_path))
        logger.info('++++ Load/Reload something from the context')


# TODO:: Fix this to monitor workspace for add/remove sub-folders in [apps, datasets]

def monitor_workspace(watched_dir, patterns, context):
    global event_handler
    if event_handler is None:
        event_handler = WatchDogHandler(patterns, context)

        observer = Observer()
        observer.schedule(event_handler, path=watched_dir, recursive=False)
        observer.start()

import logging
import random

logger = logging.getLogger(__name__)


class ActiveLearning:
    def __call__(self, request):
        strategy = request.get('strategy', 'random')
        strategy = next(iter(strategy)) if not isinstance(strategy, str) else strategy

        images = request.get('images', [])
        if not len(images):
            return None

        images.sort()
        if strategy == "first":
            image = images[0]
        elif strategy == "last":
            image = images[-1]
        else:
            image = random.choice(images)

        logger.info(f"Strategy: {strategy}; Selected Image: {image}")
        return image

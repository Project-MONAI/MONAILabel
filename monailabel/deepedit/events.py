from typing import Any

from ignite.engine.events import EventEnum, EventsList


# This class inherits/was taken from ignite.engine.Events
class DeepEditEvents(EventEnum):
    INNER_ITERATION_STARTED = "inner_iteration_started"
    INNER_ITERATION_COMPLETED = "inner_iteration_completed"

    def __or__(self, other: Any) -> "EventsList":
        return EventsList() | self | other

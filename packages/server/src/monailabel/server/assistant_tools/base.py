"""Typed tool registration and request-bound access to application services."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import ValidationError

from monailabel.core.chat import ToolCall, ToolDefinition
from monailabel.core.errors import Conflict, DomainError
from monailabel.core.models import (
    Asset,
    AssistantContext,
    AssistantReply,
    Contract,
    ModelRecord,
    Project,
    User,
)
from monailabel.server.labels import prompt_model

if TYPE_CHECKING:
    from monailabel.server.service import Services

T = TypeVar("T", bound=Contract)


def require(value: str | None, name: str) -> str:
    if not value:
        raise DomainError(f"Select {name} in the request context first.")
    return value


@dataclass
class ToolContext:
    service: "Services"
    project_id: str | None
    user: User
    context: AssistantContext
    message: str

    @property
    def project(self) -> Project:
        return self.service.store.get(Project, require(self.project_id, "a project"))

    @property
    def asset(self) -> Asset:
        asset = self.service.store.get(Asset, require(self.context.asset_id, "an asset"))
        if asset.project_id != self.project_id:
            raise DomainError("Selected asset belongs to another project.")
        if self.context.base_revision is not None and asset.revision != self.context.base_revision:
            raise Conflict(
                "The saved annotation changed while interpreting this prompt. Reload and retry."
            )
        return asset

    def model_id(self, identifier: str | None, name: str | None = None) -> str | None:
        if name:
            matches = [
                model
                for model in self.service.store.list(ModelRecord, self.project.id)
                if not model.archived and model.name.casefold() == name.strip().casefold()
            ]
            if identifier:
                matches = [m for m in matches if m.id == identifier]
            elif (
                len(matches) > 1
                and len({m.learner_id for m in matches}) == 1
                and matches[0].learner_id
            ):
                # A model name selects its latest active version, never another family.
                matches = matches[-1:]
            if len(matches) != 1:
                raise DomainError(
                    "Choose one exact model name from the available models. "
                    "If supplying a model ID as well, it must identify that same model. "
                    "No operation was started.",
                    code="invalid_model_selection",
                )
            identifier = matches[0].id
        chosen = identifier or self.context.model_id
        if chosen:
            self.service.models.get(self.project.id, chosen)
        return prompt_model(self.service.store, self.project, chosen)

    def model(self, identifier: str | None) -> ModelRecord:
        return self.service.models.get(
            self.project.id, require(self.model_id(identifier), "an annotation model")
        )


def tool_schema(model: type[Contract]) -> dict[str, Any]:
    """Keep validation constraints while omitting display-only schema titles."""

    def compact(value: Any) -> Any:
        if isinstance(value, list):
            return [compact(item) for item in value]
        if not isinstance(value, dict):
            return value
        return {
            key: (
                {name: compact(schema) for name, schema in item.items()}
                if key in {"properties", "$defs"}
                else compact(item)
            )
            for key, item in value.items()
            if key != "title"
        }

    return compact(model.model_json_schema())  # type: ignore[no-any-return]


class Empty(Contract):
    pass


@dataclass(frozen=True)
class Tool:
    definition: ToolDefinition
    arguments_model: type[Contract]
    invoke: Callable[[ToolCall], AssistantReply]
    action: str
    workspace: bool
    requires_asset: bool


class ToolRegistry:
    def __init__(self, context: ToolContext):
        self.context = context
        self.tools: dict[str, Tool] = {}

    def add(
        self,
        name: str,
        description: str,
        args: type[T],
        handler: Callable[[T], AssistantReply],
        *,
        action: str = "read",
        workspace: bool = False,
        requires_asset: bool = False,
    ) -> None:
        self.tools[name] = Tool(
            ToolDefinition(name=name, description=description, parameters=tool_schema(args)),
            args,
            lambda call: handler(args.model_validate(call.arguments)),
            action,
            workspace,
            requires_asset,
        )

    def definitions(self) -> list[ToolDefinition]:
        return [
            tool.definition
            for tool in self.tools.values()
            if (self.context.project_id or tool.workspace)
            and (self.context.context.asset_id or not tool.requires_asset)
        ]

    def execute(self, call: ToolCall) -> AssistantReply:
        tool = self.tools.get(call.name)
        if not tool or not (self.context.project_id or tool.workspace):
            raise DomainError("This tool is not available in the current workspace.")
        if self.context.project_id:
            self.context.service.auth.require(
                self.context.user, self.context.project_id, tool.action
            )
        try:
            return tool.invoke(call)
        except ValidationError as error:
            fields = ", ".join(".".join(map(str, e["loc"])) for e in error.errors())
            corrections = []
            for issue in error.errors():
                field = ".".join(map(str, issue["loc"]))
                if issue["type"] == "extra_forbidden":
                    corrections.append(f"Remove unsupported field '{field}'.")
                elif issue["type"] == "missing":
                    corrections.append(f"Supply required field '{field}'.")
                elif issue["type"] == "literal_error":
                    corrections.append(f"'{field}' must be {issue['ctx']['expected']}.")
            # Repair malformed tool schemas, not semantically invalid ranges/limits.
            # A model must not silently change a user's out-of-bounds slice request.
            repairable = all(
                e["type"] in {"literal_error", "extra_forbidden", "missing"} for e in error.errors()
            )
            raise DomainError(
                "Invalid tool arguments for "
                + fields
                + ". "
                + " ".join(corrections)
                + " No operation was started.",
                code="invalid_tool_arguments" if repairable else "invalid_request",
            ) from None

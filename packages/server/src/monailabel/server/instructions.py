"""Packaged Agent Skills with model-driven activation and bounded tool discovery.

Only trusted application resources are loaded, never uploads or the current directory.
Skill tool lists narrow model context; service authorization remains authoritative.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Literal
from xml.sax.saxutils import escape

import yaml
from pydantic import Field, ValidationError, field_validator

from monailabel.core.chat import ToolCall, ToolDefinition
from monailabel.core.errors import DomainError
from monailabel.core.models import Contract


class SkillHeader(Contract):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    description: str = Field(min_length=1, max_length=1024)
    license: str | None = None
    compatibility: str | None = Field(default=None, min_length=1, max_length=500)
    metadata: dict[str, str] = Field(default_factory=dict)
    allowed_tools: str = Field(default="", alias="allowed-tools")

    @field_validator("description")
    @classmethod
    def nonempty_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Skill description cannot be blank")
        return value


@dataclass(frozen=True)
class Skill:
    header: SkillHeader
    body: str

    @property
    def tools(self) -> set[str]:
        return set(self.header.allowed_tools.split())


def parse_skill(text: str, directory: str) -> Skill:
    """Validate bundled skills at startup; fail clearly on packaging/authoring errors."""
    sections = text.split("---\n", 2)
    if len(sections) != 3 or sections[0] or not sections[2].strip():
        raise ValueError(f"Invalid SKILL.md frontmatter or empty instructions: {directory}")
    header = SkillHeader.model_validate(yaml.safe_load(sections[1]))
    if header.name != directory:
        raise ValueError(f"Skill name must match its directory: {directory}")
    if header.metadata.get("monailabel-context") not in {"workspace", "project", "viewer"}:
        raise ValueError(
            f"Skill needs a monailabel-context of workspace, project or viewer: {directory}"
        )
    if not header.allowed_tools.strip():
        raise ValueError(f"Skill needs registered allowed-tools: {directory}")
    return Skill(header, sections[2].strip())


@lru_cache(maxsize=1)
def skills() -> tuple[Skill, ...]:
    root = files("monailabel.server").joinpath("resources/assistant/skills")
    return tuple(
        parse_skill(path.joinpath("SKILL.md").read_text(encoding="utf-8"), path.name)
        for path in sorted(root.iterdir(), key=lambda p: p.name)
        if path.is_dir() and path.joinpath("SKILL.md").is_file()
    )


@lru_cache(maxsize=4)
def coordinator_instructions(*, viewer: bool = True, project: bool = True) -> str:
    text = files("monailabel.server").joinpath("resources/assistant/AGENTS.md").read_text()
    catalog = "\n".join(
        f"<skill><name>{escape(skill.header.name)}</name>"
        f"<description>{escape(skill.header.description)}</description></skill>"
        for skill in available_skills(viewer=viewer, project=project)
    )
    return text + "\n<available_skills>\n" + catalog + "\n</available_skills>"


def available_skills(*, viewer: bool, project: bool) -> tuple[Skill, ...]:
    contexts = {"workspace"}
    if project:
        contexts.add("project")
    if viewer:
        contexts.add("viewer")
    return tuple(s for s in skills() if s.header.metadata["monailabel-context"] in contexts)


class LoadSkillArgs(Contract):
    name: str
    purpose: Literal["action", "answer"] = "action"


class SkillSession:
    """Fresh per user turn: previous workflow choices never become implicit defaults."""

    def __init__(
        self,
        definitions: list[ToolDefinition],
        *,
        viewer: bool,
        project: bool,
        inspect: Callable[[str], str] | None = None,
    ):
        self.catalog = {
            skill.header.name: skill for skill in available_skills(viewer=viewer, project=project)
        }
        self.all_tools = definitions
        self.active: set[str] = set()
        self.needs_action = False
        self.inspect = inspect

    def definitions(self) -> list[ToolDefinition]:
        exposed = {"inspect_workspace", "clarify_request"} if self.active else set()
        for name in self.active:
            exposed.update(self.catalog[name].tools)
        return [
            ToolDefinition(
                name="load_skill",
                description="Enable the action tools for a workflow and read its instructions. "
                "Tools are loaded on demand: monailabel-workspace enables creating projects; "
                "other skills enable importing, annotation, review, training or comparison. "
                "Choose a name from available_skills, then use the newly available tools.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": list(self.catalog)},
                        "purpose": {
                            "type": "string",
                            "enum": ["action", "answer"],
                            "default": "action",
                            "description": "action performs requested work (default); "
                            "answer is only for explanations or questions about a workflow.",
                        },
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            ),
            *[tool for tool in self.all_tools if tool.name in exposed],
        ]

    def load(self, call: ToolCall) -> str:
        try:
            args = LoadSkillArgs.model_validate(call.arguments)
        except ValidationError:
            raise DomainError(
                "load_skill requires a name from available_skills and optional purpose: "
                "action or answer.",
                code="invalid_tool_arguments",
            ) from None
        skill = self.catalog.get(args.name)
        if skill is None:
            raise DomainError(
                "Choose a skill name from available_skills.", code="invalid_tool_arguments"
            )
        self.needs_action = self.needs_action or args.purpose == "action"
        if args.name in self.active:
            return "This skill is already loaded. Use its tools or ask for missing information."
        content = f'<skill_instructions name="{args.name}">\n{skill.body}\n</skill_instructions>'
        if self.inspect:
            for collection in skill.header.metadata.get("monailabel-collections", "").split():
                content += (
                    "\n<workspace_data>\n"
                    + escape(self.inspect(collection))
                    + "\n</workspace_data>"
                )
        self.active.add(args.name)
        return content


def instruction_revision() -> str:
    content = coordinator_instructions() + "\n".join(
        skill.header.model_dump_json() + skill.body for skill in skills()
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


if __name__ == "__main__":
    for skill in skills():
        print(f"Valid skill: {skill.header.name}")

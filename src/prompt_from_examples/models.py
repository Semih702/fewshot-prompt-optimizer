from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["billing", "bug", "feature_request", "account_access"]
Priority = Literal["low", "medium", "high"]
LanguageCode = Literal["tr", "en"]


class TriageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category
    priority: Priority
    language: LanguageCode


class SupportExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = Field(min_length=1)
    output: TriageOutput


class SupportDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    examples: list[SupportExample] = Field(min_length=1)


class TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    output_contract: dict[str, list[str]]


class OptimizedPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: TaskDefinition
    examples: list[SupportExample] = Field(min_length=1)

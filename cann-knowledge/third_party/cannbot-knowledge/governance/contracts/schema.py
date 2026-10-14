# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Fail-closed validation for the repository's JSON Schema subset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator


DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
SUPPORTED_KEYWORDS = {
    "$schema",
    "additionalProperties",
    "enum",
    "format",
    "items",
    "maxItems",
    "minItems",
    "minLength",
    "oneOf",
    "properties",
    "required",
    "type",
    "uniqueItems",
}
SUPPORTED_TYPES = {"array": list, "object": dict, "string": str}
KEYWORD_SHAPES = {
    "additionalProperties": bool,
    "enum": list,
    "items": dict,
    "oneOf": list,
    "properties": dict,
    "required": list,
    "uniqueItems": bool,
}
TYPE_KEYWORDS = {
    "array": ("items", "maxItems", "minItems", "uniqueItems"),
    "object": ("additionalProperties", "properties", "required"),
    "string": ("format", "minLength"),
}
DATE_TIME_RE = re.compile(
    r"^(?P<prefix>\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d+))?(?P<zone>[Zz]|[+-]\d{2}:\d{2})$"
)


@dataclass(frozen=True)
class SchemaError:
    path: tuple[str | int, ...]
    message: str


class SchemaValidator:
    """Validate values against the exact JSON Schema subset used by this repo."""

    def __init__(self, schema: dict[str, Any]):
        self.schema = schema
        self._check_schema(schema, (), root=True)

    def iter_errors(self, value: Any) -> Iterator[SchemaError]:
        yield from self._validate(self.schema, value, ())

    def _check_schema(
        self,
        schema: Any,
        path: tuple[str | int, ...],
        *,
        root: bool = False,
    ) -> None:
        if not isinstance(schema, dict):
            self._invalid(path, "must be an object")
        unknown = sorted(set(schema) - SUPPORTED_KEYWORDS)
        if unknown:
            self._invalid(path, f"uses unsupported keywords: {', '.join(unknown)}")

        dialect = schema.get("$schema")
        if root and dialect != DRAFT_2020_12:
            self._invalid(path, f"must declare {DRAFT_2020_12}")
        if not root and dialect is not None:
            self._invalid(path, "may only declare $schema at the root")

        value_type = schema.get("type")
        if value_type is not None and (
            not isinstance(value_type, str) or value_type not in SUPPORTED_TYPES
        ):
            self._invalid(path, f"has unsupported type: {value_type!r}")
        for keyword, expected in KEYWORD_SHAPES.items():
            if keyword in schema and not isinstance(schema[keyword], expected):
                self._invalid(path, f"{keyword} must be {expected.__name__}")
        for keyword in ("minLength", "minItems", "maxItems"):
            value = schema.get(keyword)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                self._invalid(path, f"{keyword} must be a non-negative integer")

        enum = schema.get("enum")
        if enum is not None and (not enum or self._has_duplicates(enum)):
            self._invalid(path, "enum must be non-empty and unique")
        required = schema.get("required")
        if required is not None and (
            not all(isinstance(item, str) for item in required)
            or self._has_duplicates(required)
        ):
            self._invalid(path, "required must contain unique strings")
        properties = schema.get("properties")
        if properties is not None and not all(
            isinstance(name, str) for name in properties
        ):
            self._invalid(path, "properties keys must be strings")
        alternatives = schema.get("oneOf")
        if alternatives is not None and not alternatives:
            self._invalid(path, "oneOf must be non-empty")
        value_format = schema.get("format")
        if value_format is not None and value_format != "date-time":
            self._invalid(path, f"has unsupported format: {value_format!r}")

        for expected_type, keywords in TYPE_KEYWORDS.items():
            if any(keyword in schema for keyword in keywords):
                if value_type != expected_type:
                    self._invalid(path, f"{keywords} require type {expected_type!r}")
        if "maxItems" in schema and schema["maxItems"] < schema.get("minItems", 0):
            self._invalid(path, "maxItems is smaller than minItems")

        for name, child in (properties or {}).items():
            self._check_schema(child, (*path, "properties", name))
        if "items" in schema:
            self._check_schema(schema["items"], (*path, "items"))
        for index, child in enumerate(alternatives or []):
            self._check_schema(child, (*path, "oneOf", index))

    def _validate(
        self,
        schema: dict[str, Any],
        value: Any,
        path: tuple[str | int, ...],
    ) -> Iterator[SchemaError]:
        value_type = schema.get("type")
        type_matches = value_type is None or isinstance(
            value, SUPPORTED_TYPES[value_type]
        )
        if not type_matches:
            yield SchemaError(path, f"{value!r} is not of type {value_type!r}")

        enum = schema.get("enum")
        if enum is not None and value not in enum:
            yield SchemaError(path, f"{value!r} is not one of {enum!r}")

        alternatives = schema.get("oneOf")
        if alternatives is not None:
            matches = [
                child
                for child in alternatives
                if not list(self._validate(child, value, path))
            ]
            if not matches:
                yield SchemaError(
                    path, f"{value!r} is not valid under any of the given schemas"
                )
            elif len(matches) > 1:
                yield SchemaError(path, f"{value!r} is valid under each of {matches!r}")

        if not type_matches:
            return

        if isinstance(value, str):
            minimum = schema.get("minLength", 0)
            if len(value) < minimum:
                message = (
                    f"{value!r} should be non-empty"
                    if minimum == 1
                    else f"{value!r} is too short"
                )
                yield SchemaError(path, message)
            if schema.get("format") == "date-time" and not self._is_date_time(value):
                yield SchemaError(path, f"{value!r} is not a 'date-time'")

        if isinstance(value, list):
            if len(value) < schema.get("minItems", 0):
                yield SchemaError(path, f"{value!r} is too short")
            maximum = schema.get("maxItems")
            if maximum is not None and len(value) > maximum:
                yield SchemaError(path, f"{value!r} is too long")
            if schema.get("uniqueItems") and self._has_duplicates(value):
                yield SchemaError(path, f"{value!r} has non-unique elements")
            for index, item in enumerate(value):
                if "items" in schema:
                    yield from self._validate(schema["items"], item, (*path, index))

        if isinstance(value, dict):
            for name in schema.get("required", []):
                if name not in value:
                    yield SchemaError(path, f"{name!r} is a required property")
            properties = schema.get("properties", {})
            unexpected = [name for name in value if name not in properties]
            if schema.get("additionalProperties") is False and unexpected:
                names = ", ".join(repr(name) for name in unexpected)
                verb = "was" if len(unexpected) == 1 else "were"
                yield SchemaError(
                    path,
                    f"Additional properties are not allowed ({names} {verb} unexpected)",
                )
            for name, child in properties.items():
                if name in value:
                    yield from self._validate(child, value[name], (*path, name))

    @staticmethod
    def _has_duplicates(values: list[Any]) -> bool:
        return any(
            value == previous
            for index, value in enumerate(values)
            for previous in values[:index]
        )

    @staticmethod
    def _is_date_time(value: str) -> bool:
        match = DATE_TIME_RE.fullmatch(value)
        if match is None:
            return False
        fraction = match.group("fraction")
        normalized_fraction = (
            f".{fraction[:6].ljust(6, '0')}" if fraction is not None else ""
        )
        zone = "+00:00" if match.group("zone").lower() == "z" else match.group("zone")
        normalized = f"{match.group('prefix')}{normalized_fraction}{zone}"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return False
        return parsed.tzinfo is not None

    @staticmethod
    def _invalid(path: tuple[str | int, ...], message: str) -> None:
        location = ".".join(str(part) for part in path) or "root"
        raise ValueError(f"schema {location} {message}")

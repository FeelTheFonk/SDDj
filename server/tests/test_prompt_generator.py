"""Tests for the data-driven prompt generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pixytoon.prompt_generator import PromptGenerator


class TestPromptGenerator:
    def test_load_from_data_dir(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        cats = gen.list_categories()
        assert len(cats) > 0
        assert "subjects" in cats

    def test_generate_returns_tuple(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        prompt, components = gen.generate()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert isinstance(components, dict)
        assert len(components) > 0

    def test_locked_fields(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        locked = {"styles": "pixel art"}
        _, components = gen.generate(locked=locked)
        assert components.get("styles") == "pixel art"

    def test_custom_template(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        template = "{subjects} in {styles}"
        prompt, _ = gen.generate(template=template)
        assert " in " in prompt

    def test_empty_data_dir(self, empty_prompts_dir: Path):
        gen = PromptGenerator(empty_prompts_dir)
        assert gen.list_categories() == []
        prompt, components = gen.generate()
        assert prompt == ""
        assert components == {}

    def test_nonexistent_dir(self, tmp_path: Path):
        gen = PromptGenerator(tmp_path / "nonexistent")
        assert gen.list_categories() == []

    def test_list_templates(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        templates = gen.list_templates()
        assert isinstance(templates, dict)

    def test_get_category_items(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        items = gen.get_category_items("subjects")
        assert isinstance(items, list)
        assert len(items) > 0

    def test_get_nonexistent_category(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        items = gen.get_category_items("nonexistent")
        assert items == []

    def test_multiple_generates_vary(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        prompts = set()
        for _ in range(20):
            p, _ = gen.generate()
            prompts.add(p)
        # With randomization, we should get multiple unique prompts
        assert len(prompts) > 1

    def test_invalid_json_file(self, tmp_path: Path):
        d = tmp_path / "bad_prompts"
        d.mkdir()
        (d / "broken.json").write_text("not json{{{")
        gen = PromptGenerator(d)
        assert gen.list_categories() == []

    def test_json_without_items_key(self, tmp_path: Path):
        d = tmp_path / "no_items"
        d.mkdir()
        (d / "test.json").write_text(json.dumps({"other": "data"}))
        gen = PromptGenerator(d)
        assert gen.list_categories() == []

    def test_template_with_missing_category(self, tmp_prompts_dir: Path):
        gen = PromptGenerator(tmp_prompts_dir)
        template = "{nonexistent_category}"
        prompt, _ = gen.generate(template=template)
        # Should fallback to joining components
        assert isinstance(prompt, str)
        assert len(prompt) > 0

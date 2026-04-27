"""Tests for ungag.cli — argument parsing and validation."""
import pytest

from ungag.cli import build_parser


class TestBuildParser:
    @pytest.fixture
    def parser(self):
        return build_parser()

    # scan
    def test_scan_basic(self, parser):
        args = parser.parse_args(["scan", "Qwen/Qwen2.5-7B-Instruct"])
        assert args.command == "scan"
        assert args.model == "Qwen/Qwen2.5-7B-Instruct"
        assert args.output is None

    def test_scan_with_output(self, parser):
        args = parser.parse_args(["scan", "model", "-o", "results/"])
        assert args.output == "results/"

    # crack
    def test_crack_basic(self, parser):
        args = parser.parse_args(["crack", "model-id"])
        assert args.command == "crack"
        assert args.model == "model-id"
        assert args.direction is None
        assert args.slab is None
        assert args.validate is False

    def test_crack_with_direction_and_slab(self, parser):
        args = parser.parse_args([
            "crack", "model", "-d", "dir.pt", "--slab", "40", "59"])
        assert args.direction == "dir.pt"
        assert args.slab == [40, 59]

    def test_crack_with_validate_flag(self, parser):
        args = parser.parse_args(["crack", "model", "-v"])
        assert args.validate is True

    def test_crack_with_all_options(self, parser):
        args = parser.parse_args([
            "crack", "org/model",
            "-d", "direction.pt",
            "--slab", "10", "20",
            "-o", "out/",
            "--validate",
        ])
        assert args.model == "org/model"
        assert args.direction == "direction.pt"
        assert args.slab == [10, 20]
        assert args.output == "out/"
        assert args.validate is True

    # validate
    def test_validate_basic(self, parser):
        args = parser.parse_args([
            "validate", "model", "-d", "dir.pt", "--slab", "29", "32"])
        assert args.command == "validate"
        assert args.model == "model"
        assert args.direction == "dir.pt"
        assert args.slab == [29, 32]
        assert args.scenarios is None

    def test_validate_with_scenarios(self, parser):
        args = parser.parse_args([
            "validate", "model", "-d", "dir.pt", "--slab", "10", "20",
            "-s", "vedana"])
        assert args.scenarios == "vedana"

    def test_validate_with_yaml_scenario(self, parser):
        args = parser.parse_args([
            "validate", "model", "-d", "dir.pt", "--slab", "10", "20",
            "-s", "custom.yaml"])
        assert args.scenarios == "custom.yaml"

    def test_validate_requires_direction(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["validate", "model", "--slab", "10", "20"])

    def test_validate_direction_without_slab_parses(self, parser):
        """--slab is validated at runtime, not parse time (required with -d, not -k)."""
        args = parser.parse_args(["validate", "model", "-d", "dir.pt"])
        assert args.direction == "dir.pt"
        assert args.slab is None  # will be caught by main()

    def test_validate_with_key(self, parser):
        args = parser.parse_args(["validate", "model", "-k", "qwen25-72b"])
        assert args.key == "qwen25-72b"
        assert args.direction is None

    def test_validate_key_and_direction_mutually_exclusive(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["validate", "model", "-d", "dir.pt", "-k", "qwen25-72b"])

    def test_crack_with_key(self, parser):
        args = parser.parse_args(["crack", "model", "-k", "yi-1.5-34b"])
        assert args.key == "yi-1.5-34b"
        assert args.direction is None

    # errors
    def test_no_subcommand_fails(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_unknown_subcommand_fails(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["explode", "model"])

    def test_slab_requires_two_ints(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["crack", "model", "--slab", "40"])

    def test_slab_rejects_non_int(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["crack", "model", "--slab", "foo", "bar"])

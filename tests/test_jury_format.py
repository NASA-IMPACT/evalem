#!/usr/bin/env python3

from evalem.misc.utils import format_to_jury
from evalem._base.structures import ReferenceDTO


class References:
    class Single:
        STRS = [
            "Reference 1",
            "Reference 2",
        ]

        DICTS = [dict(text="Reference 1"), dict(text="Reference 2")]

        DTOS = [ReferenceDTO(text="Reference 1"), ReferenceDTO(text="Reference 2")]

        MIXED = [
            dict(text="Reference 1"),
            ReferenceDTO(text="Reference 2"),
        ]

    class Multi:
        STRS = [
            ["Reference 1.1", "Reference 1.2"],
            ["Reference 2.1", "Reference 2.2"],
        ]

        DICTS = [
            [dict(text="Reference 1.1"), dict(text="Reference 1.2")],
            [dict(text="Reference 2.1"), dict(text="Reference 2.2")],
        ]

        DTOS = [
            [ReferenceDTO(text="Reference 1.1"), ReferenceDTO(text="Reference 1.2")],
            [ReferenceDTO(text="Reference 2.1"), ReferenceDTO(text="Reference 2.2")],
        ]

        MIXED = [
            ["Reference 1.1", ReferenceDTO(text="Reference 1.2")],
            [dict(text="Reference 2.1"), ReferenceDTO(text="Reference 2.2")],
        ]


def test_single_strs():
    # Identity
    assert format_to_jury(References.Single.STRS) == References.Single.STRS


def test_single_dicts():
    assert format_to_jury(References.Single.DICTS) == References.Single.STRS


def test_single_dtos():
    assert format_to_jury(References.Single.DTOS) == References.Single.STRS


def test_single_mixed():
    assert format_to_jury(References.Single.MIXED) == References.Single.STRS


def test_multi_strs():
    assert format_to_jury(References.Multi.STRS) == References.Multi.STRS


def test_multi_dicts():
    assert format_to_jury(References.Multi.DICTS) == References.Multi.STRS


def test_multi_dtos():
    assert format_to_jury(References.Multi.DTOS) == References.Multi.STRS


def test_multi_mixed():
    assert format_to_jury(References.Multi.MIXED) == References.Multi.STRS


def test_single_ints():
    # right now, the jury conversion only works for only strings
    refs = [1, 2, 3, 4]
    assert format_to_jury(refs) == refs
    assert format_to_jury(refs) != list(map(str, refs))

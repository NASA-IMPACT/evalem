#!/usr/bin/env python3

from evalem.misc.utils import format_to_jury
from evalem.structures import ReferenceDTO


class References:
    class Single:
        STRS = [
            "Reference 1",
            "Reference 2",
        ]

        DICTS = [dict(text="Reference 1"), dict(text="Reference 2")]

        DTOS = [ReferenceDTO(text="Reference 1"), ReferenceDTO(text="Reference 2")]

        MIXED = [
            "Reference 1",
            dict(text="Reference 2"),
            ReferenceDTO(text="Reference 3"),
        ]


def test_single_strs():
    assert format_to_jury(References.Single.STRS) == References.Single.STRS


def test_single_dicts():
    assert format_to_jury(References.Single.DICTS) == References.Single.STRS


def test_single_dtos():
    assert format_to_jury(References.Single.DTOS) == References.Single.STRS


def test_single_mixed():
    assert format_to_jury(References.Single.MIXED) == [
        "Reference 1",
        "Reference 2",
        "Reference 3",
    ]

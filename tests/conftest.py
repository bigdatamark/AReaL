# [MODULE DOC]
# Root pytest conftest. Provides a session-scoped ip_stack fixture that probes the
# host's
# IPv4/IPv6 loopback and routing capabilities so network tests can skip appropriately.
import socket

import pytest


def _can_bind_ipv4_loopback() -> bool:
    # [FUNCTION DOC]
    # HELPER: Tries to bind a TCP socket to 127.0.0.1 on an ephemeral port; True on
    # success,
    # False on OSError.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
        return True
    except OSError:
        return False


def _can_bind_ipv6_loopback() -> bool:
    # [FUNCTION DOC]
    # HELPER: False if Python lacks IPv6 support; otherwise tries binding a TCP socket
    # to ::1
    # and returns whether it succeeded.
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


def _has_ipv4_route() -> bool:
    # [FUNCTION DOC]
    # HELPER: Connects a UDP socket to 8.8.8.8:80 (no packets sent) and returns True if
    # the
    # chosen local address is non-empty and not a 127.x address; False on OSError.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        return bool(ip) and not ip.startswith("127.")
    except OSError:
        return False


def _has_ipv6_route() -> bool:
    # [FUNCTION DOC]
    # HELPER: Same as the IPv4 probe but with a UDP socket to Google's public IPv6 DNS
    # address; True if the local address is non-empty and not ::1. False without IPv6.
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as s:
            s.connect(("2001:4860:4860::8888", 80))
            ip = s.getsockname()[0]
        return bool(ip) and ip != "::1"
    except OSError:
        return False


@pytest.fixture(scope="session")
def ip_stack() -> dict[str, bool]:
    # [FUNCTION DOC]
    # Session fixture returning a dict with bool keys ipv4_loopback, ipv6_loopback,
    # ipv4_route and ipv6_route from the probes above.
    ipv4_loopback = _can_bind_ipv4_loopback()
    ipv6_loopback = _can_bind_ipv6_loopback()
    ipv4_route = _has_ipv4_route()
    ipv6_route = _has_ipv6_route()
    return {
        "ipv4_loopback": ipv4_loopback,
        "ipv6_loopback": ipv6_loopback,
        "ipv4_route": ipv4_route,
        "ipv6_route": ipv6_route,
    }

# [MODULE DOC]
# Tests for areal/utils/network.py IPv4/IPv6 helpers. The module is loaded directly from
# its file path (avoiding the areal package import). Host-IP tests depend on the ip_stack
# fixture (dict with ipv4_route / ipv6_route booleans) and skip when the host stack does
# not match.
import importlib.util
from pathlib import Path

import pytest


def _load_network_module():
    # [FUNCTION DOC]
    # HELPER: Locates areal/utils/network.py relative to this test file, loads it as module
    # "areal_utils_network" via importlib spec, executes it and returns the module.
    # KNOWN ISSUE: the spec/loader not-None assertion runs after module_from_spec(spec) has
    # already been called, so a None spec would fail with an AttributeError first.
    root = Path(__file__).resolve().parents[1]
    path = root / "areal" / "utils" / "network.py"
    spec = importlib.util.spec_from_file_location("areal_utils_network", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_ipv4_only_environment_executes_correct_branch(ip_stack):
    # [FUNCTION DOC]
    # On IPv4-only hosts (else skip), gethostip must return an address without ":".
    if not (ip_stack["ipv4_route"] and not ip_stack["ipv6_route"]):
        pytest.skip("Not an IPv4-only host (route-level)")
    network = _load_network_module()
    assert ":" not in network.gethostip()


def test_ipv6_only_environment_executes_correct_branch(ip_stack):
    # [FUNCTION DOC]
    # On IPv6-only hosts (else skip), gethostip must return an address containing ":".
    if not (ip_stack["ipv6_route"] and not ip_stack["ipv4_route"]):
        pytest.skip("Not an IPv6-only host (route-level)")
    network = _load_network_module()
    assert ":" in network.gethostip()


def test_dual_stack_environment_executes_correct_branch(ip_stack):
    # [FUNCTION DOC]
    # On dual-stack hosts (else skip), gethostip must not return a loopback address.
    if not (ip_stack["ipv4_route"] and ip_stack["ipv6_route"]):
        pytest.skip("Not a dual-stack host (route-level)")
    network = _load_network_module()
    ip = network.gethostip()
    assert ip not in {"127.0.0.1", "::1"}


def test_split_hostport_accepts_unbracketed_ipv6():
    # [FUNCTION DOC]
    # split_hostport splits "2001:db8::1:8000" on the last colon into host and int port 8000.
    network = _load_network_module()
    host, port = network.split_hostport("2001:db8::1:8000")
    assert host == "2001:db8::1"
    assert port == 8000


def test_split_hostport_accepts_ipv4():
    # [FUNCTION DOC]
    # split_hostport parses "127.0.0.1:8000" into ("127.0.0.1", 8000).
    network = _load_network_module()
    host, port = network.split_hostport("127.0.0.1:8000")
    assert host == "127.0.0.1"
    assert port == 8000


def test_split_hostport_accepts_bracketed_ipv6():
    # [FUNCTION DOC]
    # split_hostport strips brackets from "[2001:db8::1]:8000" and returns port 8000.
    network = _load_network_module()
    host, port = network.split_hostport("[2001:db8::1]:8000")
    assert host == "2001:db8::1"
    assert port == 8000


def test_split_hostport_accepts_ipv6_with_zone_id_unbracketed():
    # [FUNCTION DOC]
    # split_hostport keeps the "%eth0" zone id in an unbracketed IPv6 host.
    network = _load_network_module()
    host, port = network.split_hostport("fe80::1%eth0:8000")
    assert host == "fe80::1%eth0"
    assert port == 8000


def test_split_hostport_accepts_ipv6_with_zone_id_bracketed():
    # [FUNCTION DOC]
    # split_hostport keeps the zone id and strips brackets for "[fe80::1%eth0]:8000".
    network = _load_network_module()
    host, port = network.split_hostport("[fe80::1%eth0]:8000")
    assert host == "fe80::1%eth0"
    assert port == 8000


def test_format_hostport_brackets_ipv6():
    # [FUNCTION DOC]
    # format_hostport wraps IPv6 hosts in brackets before appending ":port".
    network = _load_network_module()
    assert network.format_hostport("2001:db8::1", 8000) == "[2001:db8::1]:8000"


def test_format_host_for_url_idempotent():
    # [FUNCTION DOC]
    # format_host_for_url leaves already-bracketed IPv6 and plain IPv4 hosts unchanged.
    network = _load_network_module()
    assert network.format_host_for_url("[2001:db8::1]") == "[2001:db8::1]"
    assert network.format_host_for_url("127.0.0.1") == "127.0.0.1"


def test_get_loopback_ip_returns_bindable_address():
    # [FUNCTION DOC]
    # get_loopback_ip returns either "127.0.0.1" or "::1".
    network = _load_network_module()
    ip = network.get_loopback_ip()
    assert ip in {"127.0.0.1", "::1"}


def test_find_free_ports_ignores_out_of_range_excludes():
    # [FUNCTION DOC]
    # find_free_ports must return all 6 ports of range 10000-10005 when the only excluded
    # port lies outside the range (exclusions must not shrink the availability count).
    # Assumes those ports are free on the host.
    # Excluded ports outside [min_port, max_port] must not deflate the
    # availability check; all in-range ports here are free.
    network = _load_network_module()
    ports = network.find_free_ports(
        count=6, port_range=(10000, 10005), exclude_ports={50000}
    )
    assert len(ports) == 6

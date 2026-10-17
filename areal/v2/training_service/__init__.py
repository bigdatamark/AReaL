# [PACKAGE DOC]
# v2 serverless training service: a microservice stack for remote training. Workers are
# individual SPMD training processes (one per 5D-parallel rank) each behind a
# synchronous HTTP server; a data proxy fronts one parallel group and performs DP-
# partitioned dispatch; a router maps API keys to data proxies; a gateway is the
# authenticated public entry point; a controller launches and wires them; a guard forks
# worker processes.
# SPDX-License-Identifier: Apache-2.0

"""Serverless training service — microservice-based training gateway.

Workers are individual SPMD processes (one per 5D-parallel rank), each
wrapped in a synchronous HTTP server.  A data proxy orchestrates a full
5D-parallel group and provides partitioned dispatch.  A router maintains
API key → data proxy mappings.  A gateway provides the public entry
point with authentication and forwarding.
"""

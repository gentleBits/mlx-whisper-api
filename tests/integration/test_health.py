"""Integration tests for health endpoint."""

import pytest


def test_health_returns_200(client):
    """GET /health returns 200 status."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_format(client):
    """Response contains {"status": "healthy"}."""
    response = client.get("/health")
    assert response.json() == {"status": "healthy"}

"""Memory-mapped file bridge service for MT5 Strategy Tester."""
from __future__ import annotations

import json
import logging
import mmap
import time
from dataclasses import dataclass
from typing import Callable, Optional

from src.bridge.mmf_protocol import (
    BUFFER_SIZE,
    REQUEST_SIZE,
    RESPONSE_SIZE,
    ExecutionCommand,
    MarketSnapshot,
)

LOGGER = logging.getLogger("nds_bridge")


@dataclass
class BridgeConfig:
    mapping_name: str = "NDS_FLOW_BRIDGE"
    request_event: str = "NDS_FLOW_BRIDGE_REQ"
    response_event: str = "NDS_FLOW_BRIDGE_RESP"
    poll_timeout_ms: int = 250
    decision_timeout_ms: int = 500


class MMFBridgeServer:
    def __init__(self, config: BridgeConfig) -> None:
        self.config = config
        self._mapping: Optional[mmap.mmap] = None
        self._last_sequence: int = -1

    def start(self) -> None:
        LOGGER.info("Initializing MMF bridge: %s", self.config.mapping_name)
        self._mapping = mmap.mmap(-1, BUFFER_SIZE, tagname=self.config.mapping_name)
        LOGGER.info("MMF bridge initialized")

    def stop(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None

    def _read_request(self) -> Optional[MarketSnapshot]:
        if self._mapping is None:
            return None
        self._mapping.seek(0)
        payload = self._mapping.read(REQUEST_SIZE)
        try:
            snapshot = MarketSnapshot.unpack(payload)
        except Exception as exc:
            LOGGER.debug("Skipping invalid request payload: %s", exc)
            return None
        if snapshot.sequence == self._last_sequence:
            return None
        self._last_sequence = snapshot.sequence
        return snapshot

    def _write_response(self, command: ExecutionCommand) -> None:
        if self._mapping is None:
            return
        self._mapping.seek(REQUEST_SIZE)
        self._mapping.write(command.pack())

    def run(
        self,
        decision_fn: Callable[[MarketSnapshot], ExecutionCommand],
        loop_sleep: float = 0.01,
    ) -> None:
        if self._mapping is None:
            self.start()
        LOGGER.info("Bridge loop started")
        try:
            while True:
                snapshot = self._read_request()
                if snapshot:
                    LOGGER.debug("Received snapshot %s", snapshot.sequence)
                    command = decision_fn(snapshot)
                    command.sequence = snapshot.sequence
                    self._write_response(command)
                time.sleep(loop_sleep)
        except KeyboardInterrupt:
            LOGGER.info("Bridge loop interrupted")
        finally:
            self.stop()


class JsonLoggingDecisionWrapper:
    def __init__(self, decision_fn: Callable[[MarketSnapshot], ExecutionCommand]) -> None:
        self.decision_fn = decision_fn

    def __call__(self, snapshot: MarketSnapshot) -> ExecutionCommand:
        LOGGER.info(
            "Snapshot: %s",
            json.dumps(
                {
                    "symbol": snapshot.symbol,
                    "timestamp_ms": snapshot.timestamp_ms,
                    "bid": snapshot.bid,
                    "ask": snapshot.ask,
                    "spread": snapshot.spread,
                    "sequence": snapshot.sequence,
                }
            ),
        )
        command = self.decision_fn(snapshot)
        LOGGER.info("Decision: %s", json.dumps(command.to_payload()))
        return command

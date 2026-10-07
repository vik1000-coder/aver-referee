"""Offline verification of a signed Aver record. No account, no server, no trust.

This is the verify-only half of the issuer's signing path
(``stackcert_service/services/record_signing.py``), lifted so a reviewer can run
it from the free tool without importing the hosted service — and without taking
the issuer's word for anything. The issuer signs; this reads.

Four facts, and they are printed on four lines, never one:

1. **Signature.** The bytes are intact and were signed by a key the named JWKS or
   did:web document publishes. This is arithmetic, and it is all it is.
2. **Currency.** A record is current only while it is not expired, superseded,
   revoked or erased. Expiry is in the record. Revocation is *not*: it lives in a
   status list the issuer hosts, so an offline check reports it as **not checked**
   unless the user points at a status list. Reporting "valid" for a revoked record
   would be the single worst thing this file could do.
3. **Human review.** Whether a named person ratified the acceptance rules and
   signed off, as recorded at issuance.
4. **Release eligibility.** Whether this record is a basis for a release at all.
   A synthetic demonstration verifies perfectly and authorises nothing.

docs/58 requires these stay separate: a valid signature is not currency, and
neither is release eligibility.

Ed25519 needs ``cryptography``, which is the optional ``verify`` extra of the
locally built wheel (or a plain ``pip install cryptography``), so the referee
itself keeps zero dependencies. Without it, this module still *reads* a record
and reports every non-cryptographic fact, and says plainly that the signature
was not checked — it never implies a check it did not do.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ALGORITHM = "EdDSA"
KEY_TYPE = "OKP"
CURVE = "Ed25519"

MISSING_CRYPTO_MESSAGE = (
    "signature NOT CHECKED: Ed25519 verification needs the 'cryptography' package.\n"
    "         Install it with:  pip install cryptography"
)


class VerifyError(ValueError):
    """The record, the key set or the status list could not be read."""


# --------------------------------------------------------------------------- #
# compact JWS
# --------------------------------------------------------------------------- #

def b64url_decode(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except Exception as exc:
        raise VerifyError(f"not base64url: {exc}") from exc


@dataclass(frozen=True)
class Jws:
    header: dict
    payload: dict
    signing_input: bytes
    signature: bytes

    @property
    def key_id(self) -> Optional[str]:
        value = self.header.get("kid")
        return str(value) if value else None


def decode_jws(compact: str) -> Jws:
    """Split and decode a compact JWS. This verifies nothing."""

    token = compact.strip()
    parts = token.split(".")
    if len(parts) != 3:
        raise VerifyError(
            "this is not a compact JWS (expected three dot-separated parts). "
            "If you have the issuer's JSON envelope, pass its 'jws' field."
        )
    header_raw, payload_raw, signature_raw = parts
    try:
        header = json.loads(b64url_decode(header_raw))
        payload = json.loads(b64url_decode(payload_raw))
    except json.JSONDecodeError as exc:
        raise VerifyError(f"the record's header or payload is not JSON: {exc.msg}") from exc
    if not isinstance(header, dict) or not isinstance(payload, dict):
        raise VerifyError("the record's header and payload must both be JSON objects")
    return Jws(
        header=header,
        payload=payload,
        signing_input=f"{header_raw}.{payload_raw}".encode("ascii"),
        signature=b64url_decode(signature_raw),
    )


# --------------------------------------------------------------------------- #
# key material
# --------------------------------------------------------------------------- #

def public_bytes_from_jwk(jwk: dict) -> bytes:
    if str(jwk.get("kty")) != KEY_TYPE or str(jwk.get("crv")) != CURVE:
        raise VerifyError(
            f"unsupported key: kty={jwk.get('kty')!r} crv={jwk.get('crv')!r}; "
            f"this verifier reads {KEY_TYPE}/{CURVE} only"
        )
    raw = b64url_decode(str(jwk.get("x") or ""))
    if len(raw) != 32:
        raise VerifyError(f"Ed25519 public key must be 32 bytes, got {len(raw)}")
    return raw


def keys_from_jwks(document: object) -> list[dict]:
    if not isinstance(document, dict) or not isinstance(document.get("keys"), list):
        raise VerifyError("not a JWK set: expected an object with a 'keys' array")
    keys = [key for key in document["keys"] if isinstance(key, dict)]
    if not keys:
        raise VerifyError("the JWK set is empty")
    return keys


def keys_from_did_document(document: object) -> list[dict]:
    if not isinstance(document, dict):
        raise VerifyError("not a DID document")
    keys: list[dict] = []
    for method in document.get("verificationMethod") or []:
        if isinstance(method, dict) and isinstance(method.get("publicKeyJwk"), dict):
            jwk = dict(method["publicKeyJwk"])
            jwk.setdefault("kid", str(method.get("id") or "").rsplit("#", 1)[-1])
            keys.append(jwk)
    if not keys:
        raise VerifyError(
            "the DID document publishes no verificationMethod with a publicKeyJwk"
        )
    return keys


def load_document(location: str) -> object:
    """Read a JSON document from a local path or an http(s) URL the user named.

    The only network this tool ever makes is this one, and only to a location the
    user typed (docs/104 §2: no telemetry). Other schemes are refused rather than
    handed to urlopen, so a crafted ``file://`` or ``gopher://`` in a record cannot
    turn a verification into a file read.
    """

    if location.startswith(("http://", "https://")):
        request = urllib.request.Request(location, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise VerifyError(f"could not fetch {location}: {exc}") from exc
    if "://" in location:
        raise VerifyError(
            f"refusing to fetch {location!r}: only https, http and local files are read"
        )
    path = Path(location)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise VerifyError(f"cannot read {location}: {exc.strerror}") from exc
    except json.JSONDecodeError as exc:
        raise VerifyError(f"{location} is not JSON: {exc.msg}") from exc


# --------------------------------------------------------------------------- #
# the signature
# --------------------------------------------------------------------------- #

def cryptography_available() -> bool:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: F401
            Ed25519PublicKey,
        )
    except ImportError:
        return False
    return True


def verify_signature(jws: Jws, raw_public: bytes) -> bool:
    """One key, one signature. Raises if the library is missing; never guesses."""

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:  # pragma: no cover - covered by the CLI path
        raise VerifyError(MISSING_CRYPTO_MESSAGE) from exc
    try:
        Ed25519PublicKey.from_public_bytes(raw_public).verify(
            jws.signature, jws.signing_input
        )
    except InvalidSignature:
        return False
    except Exception as exc:
        raise VerifyError(f"the signature could not be checked: {exc}") from exc
    return True


def select_keys(jws: Jws, keys: list[dict]) -> list[dict]:
    """Prefer the key the header names; fall back to every published key.

    Rotation is the reason: a record signed under a retired key still verifies
    because the issuer keeps publishing it.
    """

    kid = jws.key_id
    if kid:
        matching = [key for key in keys if str(key.get("kid") or "") == kid]
        if matching:
            return matching
    return keys


# --------------------------------------------------------------------------- #
# the four facts
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    """Four independent facts about one record. They are never collapsed."""

    key_id: Optional[str] = None
    issuer: Optional[str] = None
    method_version: Optional[str] = None
    app: Optional[str] = None
    decision_point: Optional[str] = None
    example_count: Optional[int] = None
    signed_at: Optional[str] = None
    expires_at: Optional[str] = None

    signature_state: str = "not_checked"      # valid | invalid | not_checked
    signature_detail: str = ""
    currency_state: str = "not_checked"       # current | expired | revoked | not_checked
    currency_detail: str = ""
    review_state: str = "unknown"             # ratified | not_ratified | unknown
    review_detail: str = ""
    release_state: str = "unknown"            # eligible | not_eligible | unknown
    release_detail: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        """0 only when the signature checked out. Anything else is not a pass."""

        return 0 if self.signature_state == "valid" else 1

    def to_dict(self) -> dict:
        return {
            "key_id": self.key_id,
            "issuer": self.issuer,
            "method_version": self.method_version,
            "scope": {
                "app": self.app,
                "decision_point": self.decision_point,
                "example_count": self.example_count,
            },
            "signed_at": self.signed_at,
            "expires_at": self.expires_at,
            "signature": {"state": self.signature_state, "detail": self.signature_detail},
            "currency": {"state": self.currency_state, "detail": self.currency_detail},
            "human_review": {"state": self.review_state, "detail": self.review_detail},
            "release_eligibility": {"state": self.release_state, "detail": self.release_detail},
            "warnings": list(self.warnings),
        }


def _dig(payload: dict, *path: str) -> Any:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _parse_time(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def read_scope(payload: dict, result: Result) -> None:
    result.issuer = payload.get("issuer") or _dig(payload, "issuer", "id")
    subject = payload.get("credentialSubject") or payload.get("credential_subject") or payload
    result.app = _dig(subject, "app", "name") or _dig(subject, "app", "id")
    result.decision_point = (
        _dig(subject, "checkpoint", "name") or _dig(subject, "checkpoint", "type")
    )
    for path in (("evidence", "example_count"), ("evidence", "examples"),
                 ("evidence", "case_count")):
        value = _dig(subject, *path)
        if isinstance(value, int):
            result.example_count = value
            break
    result.method_version = (
        _dig(subject, "risk_policy", "method_version")
        or _dig(subject, "policy_evaluation", "method_version")
        or payload.get("method_version")
    )
    result.signed_at = payload.get("signed_at") or payload.get("issuanceDate")
    result.expires_at = payload.get("expires_at") or payload.get("expirationDate")


def read_currency(payload: dict, result: Result, *, status_list: object = None,
                  now: Optional[datetime] = None) -> None:
    """Expiry is in the record. Revocation is not, and this says so."""

    now = now or datetime.now(timezone.utc)
    expiry = _parse_time(result.expires_at)
    status = payload.get("credentialStatus") or payload.get("credential_status")

    if expiry is not None and expiry <= now:
        result.currency_state = "expired"
        result.currency_detail = f"expired {result.expires_at}"
        return

    if status_list is not None and isinstance(status, dict):
        state = _status_list_state(status, status_list)
        if state is not None:
            result.currency_state = "revoked" if state else "current"
            result.currency_detail = (
                "the status list the record names marks it revoked"
                if state else
                "the status list the record names does not mark it revoked"
            )
            if expiry is not None and not state:
                result.currency_detail += f"; not expired (expires {result.expires_at})"
            return
        result.warnings.append(
            "the status list was read but carried no entry for this record"
        )

    result.currency_state = "not_checked"
    if isinstance(status, dict) and status.get("statusListCredential"):
        result.currency_detail = (
            "NOT CHECKED offline. This record can be revoked or superseded after "
            f"issuance; its status list is {status['statusListCredential']}. "
            "Pass --status-list to check it."
        )
    else:
        result.currency_detail = (
            "NOT CHECKED offline. A signature says nothing about whether this record "
            "was later revoked, superseded or erased."
        )
    if expiry is not None:
        result.currency_detail += f" It is not expired (expires {result.expires_at})."


def _status_list_state(status: dict, document: object) -> Optional[bool]:
    """Read one index out of a status list the caller supplied. ``None`` = no entry.

    Accepts the expanded form a reviewer is likeliest to have to hand -- a map or
    list of revoked ids/indices -- rather than only the bitstring, because a
    reviewer checking one record by hand should not have to decode a gzip.
    """

    index = status.get("statusListIndex")
    identifier = status.get("id")
    if isinstance(document, dict):
        for key in ("revoked", "revokedIds", "entries", "statuses"):
            node = document.get(key)
            if isinstance(node, dict):
                for candidate in (str(index), str(identifier)):
                    if candidate in node:
                        return bool(node[candidate])
            if isinstance(node, list):
                members = {str(item) for item in node}
                if str(index) in members or str(identifier) in members:
                    return True
        if "revoked" in document or "revokedIds" in document:
            return False
    if isinstance(document, list):
        members = {str(item) for item in document}
        if str(index) in members or str(identifier) in members:
            return True
        return False
    return None


def read_review_and_eligibility(payload: dict, result: Result) -> None:
    subject = payload.get("credentialSubject") or payload.get("credential_subject") or payload
    claims = payload.get("claims") or _dig(subject, "claims") or {}

    reviewer = (
        _dig(subject, "ratified_by") or _dig(subject, "signoff", "name")
        or _dig(subject, "review", "ratified_by") or _dig(payload, "ratified_by")
    )
    if reviewer:
        result.review_state = "ratified"
        result.review_detail = f"the acceptance rules were ratified by {reviewer}"
    elif isinstance(claims, dict) and claims:
        result.review_state = "unknown"
        result.review_detail = (
            "this record does not name the person who ratified the acceptance rules"
        )
    else:
        result.review_detail = "the record carries no human-review field"

    if not isinstance(claims, dict):
        claims = {}
    if claims.get("not_deployable") or claims.get("synthetic_demonstration"):
        result.release_state = "not_eligible"
        result.release_detail = (
            "this is a demonstration over synthetic or methodology evidence: it is "
            "not customer evidence and cannot authorise a release"
        )
        return
    deployment = claims.get("deployment_use")
    if deployment == "eligible_basis":
        result.release_state = "eligible"
        result.release_detail = (
            "this record is an eligible basis for a release decision, within the "
            "scope above and only while it is current"
        )
    elif isinstance(deployment, str) and deployment:
        result.release_state = "not_eligible"
        result.release_detail = f"deployment use is recorded as {deployment!r}"
    else:
        result.release_detail = "the record states no release eligibility"


def verify_record(
    compact: str,
    keys: list[dict],
    *,
    status_list: object = None,
    now: Optional[datetime] = None,
) -> Result:
    """The whole check: decode, verify against the published keys, read the facts."""

    jws = decode_jws(compact)
    result = Result(key_id=jws.key_id)
    algorithm = str(jws.header.get("alg") or "")
    if algorithm != ALGORITHM:
        raise VerifyError(
            f"unsupported algorithm {algorithm!r}; this verifier reads {ALGORITHM} "
            "(Ed25519) only"
        )

    read_scope(jws.payload, result)
    read_currency(jws.payload, result, status_list=status_list, now=now)
    read_review_and_eligibility(jws.payload, result)

    if not cryptography_available():
        result.signature_state = "not_checked"
        result.signature_detail = MISSING_CRYPTO_MESSAGE
        return result

    candidates = select_keys(jws, keys)
    for jwk in candidates:
        if verify_signature(jws, public_bytes_from_jwk(jwk)):
            result.signature_state = "valid"
            result.key_id = str(jwk.get("kid") or jws.key_id or "")
            result.signature_detail = (
                f"the bytes are intact and were signed by key {result.key_id!r} "
                "from the key set you named"
            )
            return result
    result.signature_state = "invalid"
    result.signature_detail = (
        "NO published key in that key set verifies these bytes. Either the record "
        "was altered, or it was not signed by that issuer."
    )
    if jws.key_id and not any(str(key.get("kid") or "") == jws.key_id for key in keys):
        result.warnings.append(
            f"the record names key {jws.key_id!r}, which that key set does not publish"
        )
    return result


# --------------------------------------------------------------------------- #
# the CLI face
# --------------------------------------------------------------------------- #

_LABELS = {
    "signature": "signature",
    "currency": "currency",
    "human_review": "human review",
    "release_eligibility": "release eligibility",
}


def read_record_argument(location: str) -> str:
    """A compact JWS, straight or inside the issuer's JSON envelope."""

    text = Path(location).read_text(encoding="utf-8").strip() \
        if Path(location).exists() else location.strip()
    if text.startswith("{"):
        try:
            envelope = json.loads(text)
        except json.JSONDecodeError as exc:
            raise VerifyError(f"the record file is not JSON: {exc.msg}") from exc
        for key in ("jws", "credential_jws", "token"):
            if isinstance(envelope.get(key), str):
                return envelope[key]
        raise VerifyError("that JSON has no 'jws' field; pass the compact JWS itself")
    return text


def run_cli(args, stream) -> int:
    """``aver verify`` — four facts, four lines, and an honest exit code."""

    import json as _json

    try:
        compact = read_record_argument(args.record)
        if args.jwks and args.did:
            raise VerifyError("pass --jwks or --did, not both")
        if not args.jwks and not args.did:
            raise VerifyError(
                "say where the issuer's public key is: --jwks <url|file> or --did <url>"
            )
        document = load_document(args.jwks or args.did)
        keys = keys_from_jwks(document) if args.jwks else keys_from_did_document(document)
        status_list = load_document(args.status_list) if getattr(args, "status_list", None) \
            else None
        result = verify_record(compact, keys, status_list=status_list)
    except VerifyError as exc:
        print(f"aver: {exc}", file=stream)
        return 3

    if getattr(args, "json", False):
        _json.dump(result.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")
        return result.exit_code

    print(f"issuer          {result.issuer or '-'}", file=stream)
    print(f"key             {result.key_id or '-'}", file=stream)
    print(f"method          {result.method_version or '-'}", file=stream)
    print(f"scope           app {result.app or '-'} / decision point "
          f"{result.decision_point or '-'} / "
          f"{result.example_count if result.example_count is not None else '-'} example(s)",
          file=stream)
    print("-" * 72, file=stream)
    # docs/58: four facts, four lines. A valid signature is not currency, and
    # neither is release eligibility.
    for label, state, detail in (
        ("signature", result.signature_state, result.signature_detail),
        ("currency", result.currency_state, result.currency_detail),
        ("human review", result.review_state, result.review_detail),
        ("release eligibility", result.release_state, result.release_detail),
    ):
        print(f"{label:<20} {state.upper().replace('_', ' ')}", file=stream)
        if detail:
            for line in detail.splitlines():
                print(f"{'':<20} {line}", file=stream)
    for warning in result.warnings:
        print(f"note                 {warning}", file=stream)
    return result.exit_code

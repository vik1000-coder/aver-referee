"""``aver verify``: four facts, four lines, and never one standing in for another.

The fixtures under ``fixtures/record/`` were issued by the product's own signer
(``stackcert_service/services/record_signing.py``) and are then read here with no
service import at all — which is the point of the extraction. A test that built
its own key pair would prove the verifier agrees with itself.

The signed records are **synthetic** records made to exercise verification. Their
``example_count`` of 128 is just a value inside those records; it is not a claim
about the x10 study data (whose complete refund arm has 148 tickets).

Checking an Ed25519 signature needs ``cryptography`` (the wheel's optional
``verify`` extra). Tests that check a signature are skipped, with that reason,
when it is not importable; everything that does not need it still runs, and
``WithoutCryptographyTests`` pins what the verifier says when it is missing.
"""

from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from unittest import mock
from datetime import datetime, timezone
from pathlib import Path

from aver_cass import verify as verify_module
from aver_cli.cli import main

RECORDS = Path(__file__).resolve().parent / "fixtures" / "record"
JWKS = RECORDS / "jwks.json"
DID = RECORDS / "did.json"
RECORD = RECORDS / "record.jws"

requires_cryptography = unittest.skipUnless(
    verify_module.cryptography_available(),
    "checking an Ed25519 signature needs the optional 'cryptography' package "
    "(the wheel's `verify` extra); it is not importable here",
)


def run(argv: list[str]) -> tuple[int, str]:
    stream = io.StringIO()
    return main(argv, stream=stream), stream.getvalue()


def keys() -> list[dict]:
    return verify_module.keys_from_jwks(json.loads(JWKS.read_text(encoding="utf-8")))


def b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def tampered(record: Path = RECORD, **changes) -> str:
    """Re-encode the payload with a changed field, keeping the original signature."""

    header, payload, signature = record.read_text(encoding="utf-8").strip().split(".")
    body = json.loads(verify_module.b64url_decode(payload))
    body.update(changes)
    return ".".join([header, b64(json.dumps(body).encode("utf-8")), signature])


class DecodingTests(unittest.TestCase):
    def test_a_compact_jws_splits_into_header_payload_and_signature(self) -> None:
        jws = verify_module.decode_jws(RECORD.read_text(encoding="utf-8"))
        self.assertEqual(jws.header["alg"], "EdDSA")
        self.assertTrue(jws.key_id)
        self.assertEqual(len(jws.signature), 64)

    def test_something_that_is_not_a_jws_says_so(self) -> None:
        with self.assertRaises(verify_module.VerifyError) as caught:
            verify_module.decode_jws("hello")
        self.assertIn("compact JWS", str(caught.exception))

    def test_the_issuer_envelope_is_accepted_as_well_as_the_bare_token(self) -> None:
        envelope = verify_module.read_record_argument(str(RECORDS / "envelope.json"))
        self.assertEqual(envelope.strip(), RECORD.read_text(encoding="utf-8").strip())


class KeyMaterialTests(unittest.TestCase):
    def test_the_published_jwks_yields_a_32_byte_ed25519_key(self) -> None:
        self.assertEqual(len(verify_module.public_bytes_from_jwk(keys()[0])), 32)

    def test_a_did_document_yields_the_same_key(self) -> None:
        did_keys = verify_module.keys_from_did_document(
            json.loads(DID.read_text(encoding="utf-8")))
        self.assertEqual(verify_module.public_bytes_from_jwk(did_keys[0]),
                         verify_module.public_bytes_from_jwk(keys()[0]))

    def test_a_non_ed25519_key_is_refused_rather_than_ignored(self) -> None:
        with self.assertRaises(verify_module.VerifyError):
            verify_module.public_bytes_from_jwk({"kty": "RSA", "n": "x", "e": "AQAB"})

    def test_the_header_kid_selects_the_key_but_rotation_still_verifies(self) -> None:
        jws = verify_module.decode_jws(RECORD.read_text(encoding="utf-8"))
        retired = dict(keys()[0], kid="aver-retired-0000")
        selected = verify_module.select_keys(jws, [retired] + keys())
        self.assertEqual([key["kid"] for key in selected], [jws.key_id])
        # and with no matching kid at all, every published key is tried
        self.assertEqual(len(verify_module.select_keys(jws, [retired])), 1)

    def test_a_url_scheme_that_is_not_http_is_refused_not_opened(self) -> None:
        for location in ("file:///etc/passwd", "gopher://host/1"):
            with self.subTest(location=location):
                with self.assertRaises(verify_module.VerifyError) as caught:
                    verify_module.load_document(location)
                self.assertIn("refusing to fetch", str(caught.exception))


class FourFactsTests(unittest.TestCase):
    @requires_cryptography
    def test_a_genuine_record_verifies_offline(self) -> None:
        result = verify_module.verify_record(
            RECORD.read_text(encoding="utf-8"), keys())
        self.assertEqual(result.signature_state, "valid")
        self.assertEqual(result.method_version, "decision_policy_v3")
        self.assertEqual(result.app, "Refund agent")
        self.assertEqual(result.decision_point, "refund review")
        self.assertEqual(result.example_count, 128)
        self.assertEqual(result.exit_code, 0)

    def test_the_public_record_carries_a_count_never_the_examples(self) -> None:
        payload = verify_module.decode_jws(
            RECORD.read_text(encoding="utf-8")).payload
        self.assertIsInstance(
            payload["credentialSubject"]["evidence"]["example_count"], int)
        self.assertNotIn("examples", payload["credentialSubject"]["evidence"])

    @requires_cryptography
    def test_a_tampered_byte_fails_the_signature(self) -> None:
        result = verify_module.verify_record(
            tampered(id="rec_someone_elses"), keys())
        self.assertEqual(result.signature_state, "invalid")
        self.assertEqual(result.exit_code, 1)
        self.assertIn("altered", result.signature_detail)

    @requires_cryptography
    def test_a_key_set_that_does_not_publish_the_named_key_says_which(self) -> None:
        foreign = [dict(keys()[0], kid="aver-someone-else",
                        x=b64(bytes(32)))]
        result = verify_module.verify_record(RECORD.read_text(encoding="utf-8"), foreign)
        self.assertEqual(result.signature_state, "invalid")
        self.assertTrue(any("does not publish" in note for note in result.warnings))

    @requires_cryptography
    def test_currency_is_not_checked_offline_and_says_so_loudly(self) -> None:
        result = verify_module.verify_record(
            RECORD.read_text(encoding="utf-8"), keys())
        self.assertEqual(result.currency_state, "not_checked")
        self.assertIn("NOT CHECKED", result.currency_detail)
        # a valid signature must never be reported as currency
        self.assertNotEqual(result.signature_state, result.currency_state)

    @requires_cryptography
    def test_a_revoked_record_is_signature_valid_and_currency_revoked(self) -> None:
        status = json.loads((RECORDS / "status_list_revoked.json").read_text())
        result = verify_module.verify_record(
            RECORD.read_text(encoding="utf-8"), keys(), status_list=status)
        self.assertEqual(result.signature_state, "valid")
        self.assertEqual(result.currency_state, "revoked")

    @requires_cryptography
    def test_a_clear_status_list_reports_current(self) -> None:
        status = json.loads((RECORDS / "status_list_clear.json").read_text())
        result = verify_module.verify_record(
            RECORD.read_text(encoding="utf-8"), keys(), status_list=status)
        self.assertEqual(result.currency_state, "current")

    @requires_cryptography
    def test_an_expired_record_is_expired_even_with_a_perfect_signature(self) -> None:
        result = verify_module.verify_record(
            (RECORDS / "expired.jws").read_text(encoding="utf-8"), keys(),
            now=datetime(2026, 6, 1, tzinfo=timezone.utc))
        self.assertEqual(result.signature_state, "valid")
        self.assertEqual(result.currency_state, "expired")

    @requires_cryptography
    def test_a_demonstration_verifies_and_still_authorises_nothing(self) -> None:
        result = verify_module.verify_record(
            (RECORDS / "demonstration.jws").read_text(encoding="utf-8"), keys())
        self.assertEqual(result.signature_state, "valid")
        self.assertEqual(result.release_state, "not_eligible")
        self.assertIn("cannot authorise a release", result.release_detail)
        self.assertEqual(result.review_state, "unknown")


class CliSurfaceTests(unittest.TestCase):
    @requires_cryptography
    def test_the_four_facts_are_four_separate_lines(self) -> None:
        code, text = run(["verify", str(RECORD), "--jwks", str(JWKS)])
        self.assertEqual(code, 0)
        lines = text.splitlines()
        for label in ("signature", "currency", "human review", "release eligibility"):
            matches = [line for line in lines if line.startswith(label)]
            self.assertEqual(len(matches), 1, f"{label!r} is not on its own line")

    @requires_cryptography
    def test_a_did_document_works_as_the_key_source(self) -> None:
        code, text = run(["verify", str(RECORD), "--did", str(DID)])
        self.assertEqual(code, 0)
        self.assertIn("VALID", text)

    @requires_cryptography
    def test_json_output_keeps_the_four_facts_separate(self) -> None:
        code, text = run(["verify", str(RECORD), "--jwks", str(JWKS), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(text)
        for key in ("signature", "currency", "human_review", "release_eligibility"):
            self.assertIn("state", payload[key])
        self.assertEqual(payload["scope"]["example_count"], 128)

    def test_naming_no_key_source_is_a_usage_error(self) -> None:
        code, text = run(["verify", str(RECORD)])
        self.assertEqual(code, 3)
        self.assertIn("--jwks", text)

    def test_naming_both_key_sources_is_a_usage_error(self) -> None:
        code, _ = run(["verify", str(RECORD), "--jwks", str(JWKS), "--did", str(DID)])
        self.assertEqual(code, 3)

    @requires_cryptography
    def test_a_bad_signature_exits_nonzero(self) -> None:
        # Written to a temporary directory, never into the fixtures: the tests
        # also run from an installed wheel, whose package directory may be
        # read-only.
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "tampered.jws"
            path.write_text(tampered(id="rec_someone_elses"), encoding="utf-8")
            code, text = run(["verify", str(path), "--jwks", str(JWKS)])
        self.assertEqual(code, 1)
        self.assertIn("INVALID", text)

    def test_verify_never_imports_the_service_or_a_jwt_library(self) -> None:
        """The extraction is real: the module names the service only in prose."""

        import ast

        tree = ast.parse(Path(verify_module.__file__).read_text(encoding="utf-8"))

        def roots_of(nodes) -> set[str]:
            found: set[str] = set()
            for node in nodes:
                if isinstance(node, ast.Import):
                    found.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    found.add((node.module or "").split(".")[0])
            return found

        everywhere = roots_of(ast.walk(tree))
        for forbidden in ("stackcert_service", "stackcert", "jwt", "pydantic", "fastapi"):
            self.assertNotIn(forbidden, everywhere)
        # cryptography is imported inside the functions that need it, never at
        # module level, so the module loads and still reads a record without the
        # optional extra installed.
        self.assertIn("cryptography", everywhere)
        self.assertNotIn("cryptography", roots_of(tree.body))


class MalformedInputTests(unittest.TestCase):
    """Anything the verifier cannot read is a usage error with a reason, never a
    silent pass. None of these needs ``cryptography``."""

    def test_a_header_that_is_not_json_is_named(self) -> None:
        with self.assertRaises(verify_module.VerifyError) as caught:
            verify_module.decode_jws(f"{b64(b'not json')}.{b64(b'{}')}.{b64(bytes(64))}")
        self.assertIn("not JSON", str(caught.exception))

    def test_a_payload_that_is_json_but_not_an_object_is_refused(self) -> None:
        with self.assertRaises(verify_module.VerifyError) as caught:
            verify_module.decode_jws(f"{b64(b'{}')}.{b64(b'[1, 2]')}.{b64(bytes(64))}")
        self.assertIn("JSON objects", str(caught.exception))

    def test_an_empty_or_malformed_key_set_is_refused(self) -> None:
        for document in ({"keys": []}, {"no": "keys"}, ["not", "a", "set"]):
            with self.subTest(document=document):
                with self.assertRaises(verify_module.VerifyError):
                    verify_module.keys_from_jwks(document)
        with self.assertRaises(verify_module.VerifyError) as caught:
            verify_module.keys_from_did_document({"verificationMethod": [{"id": "x"}]})
        self.assertIn("publicKeyJwk", str(caught.exception))
        with self.assertRaises(verify_module.VerifyError):
            verify_module.keys_from_did_document("did:web:example")

    def test_an_ed25519_key_of_the_wrong_length_is_refused(self) -> None:
        with self.assertRaises(verify_module.VerifyError) as caught:
            verify_module.public_bytes_from_jwk(
                {"kty": "OKP", "crv": "Ed25519", "x": b64(bytes(31))})
        self.assertIn("32 bytes", str(caught.exception))

    def test_a_key_file_that_is_missing_or_not_json_says_which(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            missing = str(Path(work) / "absent.json")
            with self.assertRaises(verify_module.VerifyError) as caught:
                verify_module.load_document(missing)
            self.assertIn("cannot read", str(caught.exception))
            garbage = Path(work) / "garbage.json"
            garbage.write_text("{not json", encoding="utf-8")
            with self.assertRaises(verify_module.VerifyError) as caught:
                verify_module.load_document(str(garbage))
            self.assertIn("is not JSON", str(caught.exception))

    def test_an_envelope_without_a_jws_field_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            envelope = Path(work) / "envelope.json"
            envelope.write_text(json.dumps({"id": "rec_1"}), encoding="utf-8")
            with self.assertRaises(verify_module.VerifyError) as caught:
                verify_module.read_record_argument(str(envelope))
            self.assertIn("'jws'", str(caught.exception))
            envelope.write_text("{broken", encoding="utf-8")
            with self.assertRaises(verify_module.VerifyError):
                verify_module.read_record_argument(str(envelope))

    def test_the_cli_reports_an_unreadable_record_as_a_usage_error(self) -> None:
        code, text = run(["verify", "no-such-record.jws", "--jwks", str(JWKS)])
        self.assertEqual(code, 3)
        self.assertIn("compact JWS", text)


class CurrencyAndEligibilityReadingTests(unittest.TestCase):
    """The non-cryptographic facts, read straight from a payload."""

    STATUS = {"credentialStatus": {"id": "status#7", "statusListIndex": 7,
                                   "statusListCredential": "https://issuer/status"}}
    NOW = datetime(2026, 9, 23, tzinfo=timezone.utc)

    def currency(self, status_list: object, payload: dict | None = None,
                 expires: str | None = None) -> verify_module.Result:
        result = verify_module.Result(expires_at=expires)
        verify_module.read_currency(payload if payload is not None else self.STATUS,
                                    result, status_list=status_list, now=self.NOW)
        return result

    def test_every_status_list_shape_a_reviewer_might_hold_is_read(self) -> None:
        cases = (
            ([7, 9], "revoked"),                          # a bare list of indices
            (["status#7"], "revoked"),                    # a bare list of ids
            ([3], "current"),                             # a list without this record
            ({"revoked": {"7": True}}, "revoked"),        # a map of index -> flag
            ({"revoked": {"7": False}}, "current"),
            ({"revokedIds": ["status#7"]}, "revoked"),    # a named list of ids
            ({"revokedIds": ["status#8"]}, "current"),    # named list, not in it
        )
        for status_list, expected in cases:
            with self.subTest(status_list=status_list):
                self.assertEqual(self.currency(status_list).currency_state, expected)

    def test_a_status_list_with_no_entry_for_this_record_is_not_checked(self) -> None:
        result = self.currency({"something_else": []})
        self.assertEqual(result.currency_state, "not_checked")
        self.assertTrue(any("no entry" in note for note in result.warnings))
        self.assertIn("https://issuer/status", result.currency_detail)

    def test_current_and_unexpired_says_both(self) -> None:
        result = self.currency([3], expires="2027-01-01T00:00:00Z")
        self.assertEqual(result.currency_state, "current")
        self.assertIn("not expired", result.currency_detail)

    def test_a_record_without_a_status_entry_is_never_reported_current(self) -> None:
        result = self.currency([3], payload={}, expires="2027-01-01T00:00:00")
        self.assertEqual(result.currency_state, "not_checked")
        self.assertIn("not expired", result.currency_detail)

    def eligibility(self, subject: dict) -> verify_module.Result:
        result = verify_module.Result()
        verify_module.read_review_and_eligibility({"credentialSubject": subject}, result)
        return result

    def test_a_named_ratifier_and_an_eligible_basis_are_read_separately(self) -> None:
        result = self.eligibility({"ratified_by": "A. Reviewer",
                                   "claims": {"deployment_use": "eligible_basis"}})
        self.assertEqual(result.review_state, "ratified")
        self.assertIn("A. Reviewer", result.review_detail)
        self.assertEqual(result.release_state, "eligible")

    def test_any_other_deployment_use_is_not_eligible_and_says_what_it_was(self) -> None:
        result = self.eligibility({"claims": {"deployment_use": "advisory_only"}})
        self.assertEqual(result.release_state, "not_eligible")
        self.assertIn("advisory_only", result.release_detail)
        self.assertEqual(result.review_state, "unknown")

    def test_a_record_that_states_nothing_claims_nothing(self) -> None:
        result = self.eligibility({})
        self.assertEqual((result.review_state, result.release_state),
                         ("unknown", "unknown"))
        self.assertIn("no human-review field", result.review_detail)
        self.assertIn("no release eligibility", result.release_detail)


class WithoutCryptographyTests(unittest.TestCase):
    """Without the optional extra the verifier still reads, and never pretends.

    Runs everywhere: the missing library is simulated, so this is pinned even on
    a machine that has ``cryptography`` installed.
    """

    def setUp(self) -> None:
        patcher = mock.patch.object(verify_module, "cryptography_available",
                                    return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_signature_is_not_checked_and_the_other_facts_are_still_read(self) -> None:
        result = verify_module.verify_record(RECORD.read_text(encoding="utf-8"), keys())
        self.assertEqual(result.signature_state, "not_checked")
        self.assertIn("NOT CHECKED", result.signature_detail)
        self.assertIn("pip install cryptography", result.signature_detail)
        # not checked is never a pass
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.decision_point, "refund review")
        self.assertIsInstance(result.example_count, int)

    def test_the_cli_says_so_instead_of_printing_valid(self) -> None:
        code, text = run(["verify", str(RECORD), "--jwks", str(JWKS)])
        self.assertEqual(code, 1)
        self.assertIn("NOT CHECKED", text)
        self.assertNotIn("VALID", text.replace("INVALID", ""))


if __name__ == "__main__":
    unittest.main()

"""Offline handoff to a customer's labeling agent. No provider calls or approval authority.

The request blinds option names and prices, strips credential fields, and binds the response to the exact export.
Review the request before sending its messages to a model: content can still contain personal information.
"""
from __future__ import annotations

import json
import re

from aver_policy.decision import digest, need
from aver_policy import log_reader as lr
from aver_policy import log_formats

SECRET = re.compile(r'authorization|cookie|password|secret|token|api.?key|credential|cost|price|tolerance|budget|billing|model|provider|variant|configuration', re.I)


def scrub(value):
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if not SECRET.search(k)}
    if isinstance(value, list): return [scrub(v) for v in value]
    if isinstance(value, str):
        return re.sub(r'(?i)\bBearer\s+\S+|\bsk-[A-Za-z0-9_-]{12,}', '[redacted]', value)
    return value


def records_from(logs, log_format=None, prices=None):
    hint = None
    if isinstance(logs, str): logs, hint = log_formats.load_text(logs)
    records = lr.normalise_logs(log_formats.adapt(logs, log_format, prices, hint)['records'])
    need(0 < len(records) <= 5000, 'invalid_label_request', 'logs', 'Use between 1 and 5,000 logged runs per labeling handoff.')
    pairs = [(r['candidate_id'], r['case_id']) for r in records]
    need(all(all(isinstance(v, str) and lr.valid_label(v) for v in p) for p in pairs) and len(set(pairs)) == len(pairs),
         'invalid_label_request', 'logs', 'Every run needs a unique case and option pair.')
    return records


def make_request(logs, model, include_content=False, log_format=None, prices=None):
    need(isinstance(model, str) and 0 < len(model.strip()) <= 120, 'invalid_label_request', 'model', 'Name the exact model your labeling agent will use.')
    recs = records_from(logs, log_format, prices)
    candidates = {v: f'option_{i}' for i, v in enumerate(sorted({r['candidate_id'] for r in recs}))}
    cases = {v: f'case_{i}' for i, v in enumerate(sorted({r['case_id'] for r in recs}))}
    records = []
    for r in recs:
        row = {'candidate_id': candidates[r['candidate_id']], 'case_id': cases[r['case_id']]}
        if include_content:
            row.update({k: scrub(r.get(k)) for k in ('input', 'output', 'expected', 'tool_calls', 'error', 'meta')})
        records.append(row)
    value = {'schema': 'aver-label-request/v1', 'source_digest': digest(recs), 'model': model.strip(), 'content_included': include_content,
             'candidates': {v: k for k, v in candidates.items()}, 'cases': {v: k for k, v in cases.items()},
             'messages': [{'role': 'system', 'content': lr.LLM_READER_SYSTEM_PROMPT + '\nIf no content supports a judgment, mark the run unsure. Strata use meta fields only; leave field empty to use the offline reader\'s split.'},
                          {'role': 'user', 'content': json.dumps({'records': records, 'standard_kinds': lr.blind_taxonomy(lr.standard_taxonomy())}, sort_keys=True)}],
             'response_schema': lr.LLM_READER_RESPONSE_SCHEMA}
    value['request_digest'] = digest(value)
    return value


def obj(value, required, optional=()):
    need(isinstance(value, dict) and set(required) <= set(value) and set(value) <= set(required) | set(optional),
         'invalid_model_response', 'response', 'Missing or unexpected response fields.')


def text(value, limit):
    need(isinstance(value, str) and 0 < len(value.strip()) <= limit, 'invalid_model_response', 'response', 'A response sentence is missing or too long.')


class ResponseReader(lr.LLMReader):
    name = 'model-response'
    version = '1'

    def __init__(self, request, envelope):
        obj(request, ['schema', 'source_digest', 'model', 'content_included', 'candidates', 'cases', 'messages', 'response_schema', 'request_digest'])
        need(request['schema'] == 'aver-label-request/v1' and request['request_digest'] == digest({k: v for k, v in request.items() if k != 'request_digest'}),
             'invalid_label_request', 'request', 'The labeling request changed; make a new request.')
        need(isinstance(request['model'], str) and 0 < len(request['model']) <= 120 and type(request['content_included']) is bool
             and isinstance(request['source_digest'], str) and re.fullmatch(r'[a-f0-9]{64}', request['source_digest'])
             and all(isinstance(request[k], dict) and request[k] and all(isinstance(v, str) and lr.valid_label(v) for v in request[k].values()) for k in ('cases', 'candidates')),
             'invalid_label_request', 'request', 'Use a labeling request generated from your export.')
        obj(envelope, ['schema', 'request_digest', 'model', 'response'])
        need(envelope['schema'] == 'aver-label-response/v1' and envelope['request_digest'] == request['request_digest'] and envelope['model'] == request['model'],
             'label_response_mismatch', 'response', 'The response must name this request and the selected model.')
        self.request, self.response = request, envelope['response']
        self.provenance = {'request_digest': request['request_digest'], 'response_digest': digest(envelope), 'model': envelope['model'],
                           'model_identity': 'reported_by_customer_agent', 'content_included': request['content_included']}

    def prepare(self, records):
        need(digest(records) == self.request['source_digest'], 'label_source_changed', 'logs', 'These logs differ from the labeling request. Make a new request.')
        r = self.response
        obj(r, ['purpose', 'deployment', 'strata', 'kinds', 'labels']); text(r['purpose'], 200)
        obj(r['deployment'], ['mode', 'why']); text(r['deployment']['why'], 300)
        need(r['deployment']['mode'] in ('normal', 'agentic'), 'invalid_model_response', 'deployment', 'Choose normal or agentic.')
        obj(r['strata'], ['field', 'why']); text(r['strata']['why'], 300)
        need(isinstance(r['strata']['field'], str) and (not r['strata']['field'] or re.fullmatch(r'meta\.[A-Za-z0-9_.]+', r['strata']['field'])),
             'invalid_model_response', 'strata', 'Use a case-level meta field or leave it empty.')
        kinds = r['kinds']
        need(isinstance(kinds, dict) and 0 < len(kinds) <= 64, 'invalid_model_response', 'kinds', 'Propose between 1 and 64 mistake kinds.')
        for k, v in kinds.items():
            need(isinstance(k, str) and re.fullmatch(r'[a-z][a-z0-9_]{0,63}', k), 'invalid_model_response', 'kinds', 'Kind keys must be ASCII snake_case.')
            obj(v, ['label', 'severity', 'standard', 'evidence']); text(v['label'], 120)
            need(isinstance(v['standard'], str) and v['standard'] in lr.STANDARD_TAXONOMY and v['severity'] == lr.STANDARD_TAXONOMY[v['standard']]['severity'],
                 'invalid_model_response', 'kinds', 'A kind must keep its standard parent and severity.')
            need(isinstance(v['evidence'], list) and 0 < len(v['evidence']) <= 5 and all(isinstance(x, str) and x in self.request['cases'] for x in v['evidence']),
                 'invalid_model_response', 'evidence', 'Each kind needs up to five case IDs from this request.')
        need(isinstance(r['labels'], list), 'invalid_model_response', 'labels', 'Return one label per requested run.')
        self.labels = {}
        for lab in r['labels']:
            obj(lab, ['case_id', 'candidate_id', 'mistake', 'rationale'], ['unsure']); text(lab['rationale'], 200)
            need(isinstance(lab['case_id'], str) and isinstance(lab['candidate_id'], str) and lab['case_id'] in self.request['cases'] and lab['candidate_id'] in self.request['candidates'],
                 'invalid_model_response', 'labels', 'A label names a case or option outside this request.')
            need(type(lab.get('unsure', False)) is bool and (lab['mistake'] is None or isinstance(lab['mistake'], str) and lab['mistake'] in kinds)
                 and not (lab.get('unsure') and lab['mistake'] is not None), 'invalid_model_response', 'labels', 'Use a known mistake kind, or null; unsure labels must have mistake null.')
            key = (self.request['candidates'][lab['candidate_id']], self.request['cases'][lab['case_id']])
            need(key not in self.labels, 'invalid_model_response', 'labels', 'Repeated labels are refused.')
            self.labels[key] = lab
        need(set(self.labels) == {(x['candidate_id'], x['case_id']) for x in records}, 'invalid_model_response', 'labels', 'Return exactly one label for every result; no missing or invented results.')

    def describe(self, records): return {'purpose': self.response['purpose'], 'deployment': self.response['deployment']}

    def propose_taxonomy(self, records, standard=None):
        return {'version': (standard or lr.standard_taxonomy())['version'], 'kinds': {
            k: {key: v[key] for key in ('label', 'severity', 'standard')} | {'price_usd': None, 'tolerance_per_month': None}
            for k, v in self.response['kinds'].items()}}

    def label(self, record, taxonomy): return self.labels[(record['candidate_id'], record['case_id'])]

    def propose_strata(self, records, labels=None):
        field = self.response['strata']['field']
        if not field: return super().propose_strata(records, labels)
        cases = {}
        for row in records:
            value = lr._strat_value(row, field)
            need(value != 'None' and (row['case_id'] not in cases or cases[row['case_id']] == value),
                 'invalid_model_response', 'strata', 'The split must exist and be the same for a case on every option.')
            cases[row['case_id']] = value
        values = sorted(set(cases.values()))
        need(1 <= len(values) <= 6 and all(lr.valid_label(v) for v in values), 'invalid_model_response', 'strata', 'Use at most six named kinds of work.')
        self.strata_rationale = self.response['strata']['why']
        return {v: {'weight': sum(x == v for x in cases.values()) / len(cases), 'failure_limit': None, 'how': field + ' as logged'} for v in values}

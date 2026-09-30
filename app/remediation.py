"""License remediation planning for ``status == "conflict"`` assessments.

Given a conflict assessment and, per adjustable component, a list of
candidate replacement license expressions (each with a stable id and a
non-negative cost), find the globally optimal remediation -- or prove
none exists -- using **only the normalized request and rules snapshot
persisted on that assessment**.  The currently active ruleset is never
read: a row whose snapshot cannot be verified is refused with HTTP 409.

Pipeline:

1. normalize/validate the candidate list.  Structural envelope problems
   are HTTP 400; a candidate that is otherwise well-formed but illegal
   (duplicate id within a component, reference to a component that is not
   part of the assessment, unparseable expression, license/exception
   without a rule, negative/non-finite cost) makes the whole request
   HTTP 422 -- no partial conclusion is ever produced.  Components
   without candidates keep their original expression at zero cost
   (the implicit "keep" option, ``candidate_id: null`` in the response);
2. for every component, options = [keep] + its candidates; an option
   that changes the expression is flagged via the parsed OR-alternative
   leaf sets (rendering differences that preserve the same alternatives
   are not a change);
3. solve globally over option assignments.

   The releasable optimum minimizes
       (total cost, changed-component count,
        sequence of (component id, candidate id) in component-id order);
   inner OR branches are then fixed by the ordinary deterministic
   re-assessment, so equal candidate choices always yield one identical
   result.  When no releasable assignment exists, the unique best failing
   assignment minimizes
       (conflicting-component count, total cost,
        sequence of (component id, candidate id)).

   Small option spaces are enumerated exhaustively (each assignment is
   re-assessed with the pinned rules, so the recomputed adopted branches,
   obligations and conflict explanation are always real solver output).
   Larger spaces use a complete branch-and-bound over the flattened
   (option x OR-branch) space for the releasable search; if that proves
   every selection conflicts, the best failing assignment is found with
   a complete outer enumeration -- only provably non-improving subtrees
   are skipped, never an approximation (``selection_mode`` reports
   ``exhaustive`` / ``branch-and-bound``).

All costs are exact rationals internally; request ordering (candidates,
components) never influences the result, and ``remediation_hash`` is the
SHA-256 of the canonical response without the hash field.
"""

import math
from fractions import Fraction
from itertools import product

from .common import ClientError, stable_hash
from .expression import ExpressionError, alternatives, parse
from .fulfillment import UnprocessableError
from .solver import _BranchBound, _deps_first_order, assess_normalized

# Cap on the number of option assignments enumerated exhaustively; beyond
# it the flattened branch-and-bound path takes the releasable search.
MAX_OPTION_ASSIGNMENTS = 4096

_CANDIDATE_FIELDS = {"id", "component", "license", "cost"}


class NoReleaseError(Exception):
    """HTTP 409 carrying the structured best-failing-assignment payload."""

    def __init__(self, payload):
        super().__init__(payload["error"])
        self.payload = payload


# ---------------------------------------------------------------------------
# canonical expression rendering / option signatures
# ---------------------------------------------------------------------------

def _render(node):
    """Render a parsed AST back to a canonical SPDX-style expression.

    AND binds tighter than OR; same-precedence children are flattened and
    sorted, so semantically equal expressions render identically.
    """
    kind = node[0]
    if kind == "license":
        return "%s WITH %s" % (node[1], node[2]) if node[2] else node[1]
    if kind == "and":
        parts = [
            "(" + _render(ch) + ")" if ch[0] == "or" else _render(ch)
            for ch in node[1]
        ]
        return " AND ".join(sorted(parts))
    parts = [
        "(" + _render(ch) + ")" if ch[0] == "and" else _render(ch)
        for ch in node[1]
    ]
    return " OR ".join(sorted(parts))


def _signature(alts):
    """Frozen identity of an expression: the set of its OR alternatives,
    each the set of (license, exception) leaves it AND-combines."""
    return frozenset(
        frozenset((lid, exc or "") for lid, exc in alt) for alt in alts
    )


def _expression_alts(ast):
    return sorted(
        {
            tuple(sorted({(lid, exc or "") for lid, exc in alt}))
            for alt in alternatives(ast)
        }
    )


def _walk_licenses(ast):
    """Yield license AST nodes (iterative, mirrors expression._walk)."""
    stack = [ast]
    while stack:
        node = stack.pop()
        if node[0] == "license":
            yield node
        else:
            stack.extend(reversed(node[1]))


# ---------------------------------------------------------------------------
# candidate normalization / validation
# ---------------------------------------------------------------------------

def _cost(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClientError("%s.cost must be a non-negative number" % where)
    if not math.isfinite(value) or value < 0:
        raise UnprocessableError(
            "%s.cost must be a non-negative finite number" % where
        )
    return Fraction(value)


def normalize_candidates(body, comp_ids, rules):
    """Validate and canonicalize the submitted candidate list.

    Structural problems raise :class:`ClientError` (400); illegal
    candidates raise :class:`UnprocessableError` (422), collected into
    one deterministic, order-independent error.  Returns
    ``(by_component, echo)``: ``by_component`` maps component id to its
    candidate records sorted by candidate id; ``echo`` is the normalized
    candidate list echoed in the response.
    """
    if not isinstance(body, dict):
        raise ClientError("request body must be a JSON object")
    raw_list = body.get("candidates", [])
    if not isinstance(raw_list, list):
        raise ClientError("candidates must be an array")

    ids = set(comp_ids)
    license_rules = rules["licenses"]
    exception_rules = rules.get("exceptions", {})

    records = []
    for i, raw in enumerate(raw_list):
        where = "candidates[%d]" % i
        if not isinstance(raw, dict):
            raise ClientError("%s must be an object" % where)
        unknown = sorted(set(raw) - _CANDIDATE_FIELDS)
        if unknown:
            raise ClientError("%s: unknown fields %s" % (where, unknown))
        cand_id = raw.get("id")
        if not isinstance(cand_id, str) or not cand_id.strip():
            raise ClientError("%s.id must be a non-empty string" % where)
        component = raw.get("component")
        if not isinstance(component, str) or not component.strip():
            raise ClientError("%s.component must be a non-empty string" % where)
        license_text = raw.get("license")
        if not isinstance(license_text, str) or not license_text.strip():
            raise ClientError("%s.license must be a non-empty string" % where)
        cost = _cost(raw.get("cost", 0), where)
        records.append(
            {
                "candidate_id": cand_id.strip(),
                "component": component.strip(),
                "expression": license_text.strip(),
                "cost": cost,
            }
        )

    # Semantic validation: every failure is reported together (422), in a
    # deterministic order that does not depend on the submitted ordering.
    problems = []
    for rec in records:
        where = "candidate %r for component %r" % (
            rec["candidate_id"], rec["component"],
        )
        if rec["component"] not in ids:
            problems.append(
                "%s references a component that is not part of this assessment"
                % where
            )
            continue
        try:
            ast = parse(rec["expression"])
        except ExpressionError as exc:
            problems.append("%s: invalid license expression (%s)" % (where, exc))
            continue
        missing_licenses = set()
        missing_exceptions = set()
        for node in _walk_licenses(ast):
            if node[1] not in license_rules:
                missing_licenses.add(node[1])
            if node[2] and node[2] not in exception_rules:
                missing_exceptions.add(node[2])
        missing = sorted(missing_licenses | missing_exceptions)
        if missing:
            problems.append(
                "%s: no rule for license/exception id(s) %s"
                % (where, sorted(set(missing)))
            )
            continue
        rec["ast"] = ast
        rec["canonical"] = _render(ast)
        rec["alts"] = _expression_alts(ast)
    if problems:
        raise UnprocessableError("; ".join(sorted(problems)))

    by_component = {}
    for rec in records:
        by_component.setdefault(rec["component"], []).append(rec)

    # Candidate ids are stable per component (the id space is scoped to
    # the component they adjust).
    dup_problems = []
    for cid, recs in by_component.items():
        seen = set()
        for rec in recs:
            if rec["candidate_id"] in seen:
                dup_problems.append(
                    "duplicate candidate id %r for component %r"
                    % (rec["candidate_id"], cid)
                )
            seen.add(rec["candidate_id"])
    if dup_problems:
        raise UnprocessableError("; ".join(sorted(dup_problems)))

    for cid in by_component:
        by_component[cid].sort(key=lambda r: r["candidate_id"])
    echo = [
        {
            "component": rec["component"],
            "candidate_id": rec["candidate_id"],
            "license": rec["canonical"],
            "cost": _num(rec["cost"]),
        }
        for cid in sorted(by_component)
        for rec in by_component[cid]
    ]
    return by_component, echo


def _num(frac):
    """Render an exact rational cost as a JSON number (int when integral)."""
    if frac.denominator == 1:
        return frac.numerator
    return float(frac)


# ---------------------------------------------------------------------------
# per-component options
# ---------------------------------------------------------------------------

def _build_options(norm, by_component):
    """Per component: the ordered list of selectable options.

    Option 0 is always the zero-cost "keep" of the persisted expression;
    candidates follow sorted by candidate id.  Each option carries its
    parsed OR alternatives, canonical expression, exact cost and whether
    adopting it semantically changes the component's expression.
    """
    options = {}
    for c in norm["components"]:
        cid = c["id"]
        ast = parse(c["license"])
        keep_alts = _expression_alts(ast)
        keep_sig = _signature(keep_alts)
        opts = [
            {
                "kind": "keep",
                "candidate_id": None,
                "canonical": _render(ast),
                "alts": keep_alts,
                "cost": Fraction(0),
                "changed": False,
            }
        ]
        for rec in by_component.get(cid, []):
            opts.append(
                {
                    "kind": "candidate",
                    "candidate_id": rec["candidate_id"],
                    "canonical": rec["canonical"],
                    "alts": rec["alts"],
                    "cost": rec["cost"],
                    "changed": _signature(rec["alts"]) != keep_sig,
                }
            )
        options[cid] = opts
    return options


def _choice_sequence(comp_ids, chosen):
    """(component id, candidate id) sequence in component-id order; the
    keep option serializes as the empty string, smaller than any real
    non-empty candidate id."""
    return tuple((cid, chosen[cid]["candidate_id"] or "") for cid in comp_ids)


def _modified_norm(norm, adjusted, options, picks):
    """A copy of the normalized request with each adjusted component's
    license replaced by the chosen option's canonical expression.

    Components without submitted candidates are untouched.  Canonical
    re-rendering never changes a solver result (OR alternatives are
    normalized into sorted leaf sets before evaluation), so "keep"
    choices are semantically -- and result-wise -- the original.
    """
    chosen_license = {
        cid: options[cid][picks[cid]]["canonical"] for cid in adjusted
    }
    components = []
    for c in norm["components"]:
        c2 = dict(c)
        if c["id"] in chosen_license:
            c2["license"] = chosen_license[c["id"]]
        components.append(c2)
    modified = dict(norm)
    modified["components"] = components
    return modified


# ---------------------------------------------------------------------------
# exhaustive solver over option assignments
# ---------------------------------------------------------------------------

def _solve_exhaustive(norm, rules, adjusted, options):
    """Enumerate every option assignment and score the recomputed result.

    Returns ``(release_best, failure_best)``; each best is
    ``(key, picks, result)`` or ``None``.  Every assignment is assessed
    with the pinned rules snapshot, so the stored result is an ordinary
    solver output -- adopted branches and obligations are never rebuilt
    by hand.
    """
    release_best = None
    failure_best = None
    ordered_adjusted = sorted(adjusted)
    for combo in product(*[options[cid] for cid in ordered_adjusted]):
        chosen = dict(zip(ordered_adjusted, combo))
        picks = {cid: options[cid].index(opt) for cid, opt in chosen.items()}
        total = sum((opt["cost"] for opt in combo), Fraction(0))
        changed = sum(1 for opt in combo if opt["changed"])
        result = assess_normalized(
            _modified_norm(norm, ordered_adjusted, options, picks), rules
        )
        seq = _choice_sequence(ordered_adjusted, chosen)
        if result["status"] == "ok":
            key = (total, changed, seq)
            if release_best is None or key < release_best[0]:
                release_best = (key, chosen, result)
        else:
            n_conf = len(result.get("conflicting_components", []))
            key = (n_conf, total, seq)
            if failure_best is None or key < failure_best[0]:
                failure_best = (key, chosen, result)
    return release_best, failure_best


# ---------------------------------------------------------------------------
# flattened branch-and-bound for large option spaces
# ---------------------------------------------------------------------------

def _merge_alternatives(options):
    """Flatten options into one sorted leaf-alternative list per component.

    Returns ``(alts, alt_meta, scale, option_index)`` where
    ``alt_meta[cid][leaf_index]`` is ``(candidate_id, scaled_cost,
    changed)`` and ``option_index[cid][candidate_id]`` resolves an id
    ("" for keep) back to its option record.

    When several options contain the same OR-branch leaf set it collapses
    onto the option that wins for that leaf under the per-component order
    (keep, then cost, then changed, then candidate id): the objective
    separates additively over components, so this never moves the global
    optimum or its unique tie-break.
    """
    scale = 1
    for opts in options.values():
        for opt in opts:
            d = opt["cost"].denominator
            scale = scale * d // math.gcd(scale, d)

    def scaled(opt):
        return opt["cost"].numerator * (scale // opt["cost"].denominator)

    alts, alt_meta, option_index = {}, {}, {}
    for cid, opts in options.items():
        winners = {}
        index = {}
        for opt in opts:
            if opt["candidate_id"] is None:
                index[""] = opt
            else:
                index[opt["candidate_id"]] = opt
            sort_key = (
                0 if opt["kind"] == "keep" else 1,
                scaled(opt),
                1 if opt["changed"] else 0,
                opt["candidate_id"] or "",
            )
            for leaf in opt["alts"]:
                if leaf not in winners or sort_key < winners[leaf][0]:
                    winners[leaf] = (sort_key, opt)
        leafs = sorted(winners)
        alts[cid] = leafs
        meta = {}
        for i, leaf in enumerate(leafs):
            opt = winners[leaf][1]
            meta[i] = (
                opt["candidate_id"] or "",
                scaled(opt),
                bool(opt["changed"]),
            )
        alt_meta[cid] = meta
        option_index[cid] = index
    return alts, alt_meta, scale, option_index


class _RemediationSearch(_BranchBound):
    """Branch-and-bound over the flattened (option x OR-branch) space.

    Reuses the generic guaranteed-conflict lower bound, incremental
    presence/spread state and exact leaf evaluation of
    :class:`~app.solver._BranchBound`.  The objective is
    ``(conflict count, total cost, changed count, (component, candidate)
    sequence)``; its first component is 0 exactly for releasable
    selections.  A subtree is pruned only when every completion is
    provably unable to improve the incumbent, so the outcome is identical
    to exhaustive enumeration.
    """

    def __init__(self, norm, rules, alts, alt_meta, adjusted, option_index):
        components = {c["id"]: c for c in norm["components"]}
        comp_ids = sorted(components)
        edges = [(d["from"], d["to"]) for d in norm["dependencies"]]
        super().__init__(
            components, comp_ids, edges, rules, norm["distribution"], alts
        )
        self._alt_meta = alt_meta
        self._adjusted = set(adjusted)
        self._option_index = option_index

    def find_release(self):
        """Return the chosen option per adjusted component for the optimal
        conflict-free selection, or ``None`` when every flattened
        selection conflicts."""
        comp_ids = self._comp_ids
        alts = self._alts
        adjusted = sorted(self._adjusted)
        order = _deps_first_order(comp_ids, self._edges)
        n = len(order)

        # Cheapest/least-changing alternative first.  Search order never
        # changes the optimum, only how early the incumbent appears.
        value_order = {}
        for cid in comp_ids:
            if cid in self._adjusted:
                value_order[cid] = sorted(
                    range(len(alts[cid])),
                    key=lambda i: (
                        self._alt_meta[cid][i][1],
                        1 if self._alt_meta[cid][i][2] else 0,
                        self._alt_meta[cid][i][0],
                        alts[cid][i],
                    ),
                )
            else:
                value_order[cid] = list(range(len(alts[cid])))

        best = None  # [(nc, cost, changed, seq), assigned-index snapshot]
        cur_cost = 0
        cur_changed = 0
        tags = {}  # adjusted cid -> candidate id ("" for keep)

        def seq_with_defaults():
            # Unassigned adjusted components may still take any option,
            # whose smallest candidate id is "" (keep, cost 0): this tuple
            # is the lexicographically smallest completion sequence and
            # 0/0 are the smallest completion cost/changed contributions.
            return tuple((cid, tags.get(cid, "")) for cid in adjusted)

        # Frames mirror _BranchBound.find_best: a frame carries the mark
        # and the cost-meta of its PARENT assignment, so exhausting a
        # frame undoes exactly what the parent applied.
        frames = [[0, iter(value_order[order[0]]), None, None]]
        while frames:
            depth, it, _pmark, _pmeta = frames[-1]
            try:
                idx = next(it)
            except StopIteration:
                _, _, pmark, pmeta = frames.pop()
                if pmeta is not None:
                    pcid, pidx = pmeta
                    _, pcost, pchanged = self._alt_meta[pcid][pidx]
                    cur_cost -= pcost
                    cur_changed -= 1 if pchanged else 0
                    del tags[pcid]
                if pmark is not None:
                    self._undo(pmark)
                continue
            cid = order[depth]
            mark = self._assign(cid, idx)
            if cid in self._adjusted:
                meta = self._alt_meta[cid][idx]
                cand, cost, changed = meta
                cur_cost += cost
                cur_changed += 1 if changed else 0
                tags[cid] = cand
            else:
                meta = None

            prune = False
            if best is not None:
                bnc, bcost, bchg, bseq = best[0]
                lb = self._lb  # guaranteed conflicts: <= every completion
                if lb > bnc:
                    prune = True
                elif lb == bnc:
                    # Remaining components can keep for free, so current
                    # cost/changed are exact lower bounds.
                    if (cur_cost, cur_changed) > (bcost, bchg):
                        prune = True
                    elif (cur_cost, cur_changed) == (bcost, bchg):
                        if seq_with_defaults() > bseq:
                            prune = True
            if prune:
                if meta is not None:
                    cur_cost -= meta[1]
                    cur_changed -= 1 if meta[2] else 0
                    del tags[cid]
                self._undo(mark)
                continue

            if depth + 1 < n:
                frames.append(
                    [
                        depth + 1,
                        iter(value_order[order[depth + 1]]),
                        mark,
                        (cid, idx) if meta is not None else None,
                    ]
                )
                continue

            _, ev, _ = self._evaluate_current()
            key = (
                len(ev["conflicts"]), cur_cost, cur_changed, seq_with_defaults()
            )
            if best is None or key < best[0]:
                best = [key, dict(self._assigned)]
            if meta is not None:
                cur_cost -= meta[1]
                cur_changed -= 1 if meta[2] else 0
                del tags[cid]
            self._undo(mark)

        if best is None or best[0][0] > 0:
            return None
        chosen = {}
        for cid in adjusted:
            cand = self._alt_meta[cid][best[1][cid]][0]
            chosen[cid] = self._option_index[cid][cand]
        return chosen


def _solve_branch_bound(norm, rules, adjusted, options):
    """Large-space path.

    The flattened complete branch-and-bound decides whether any
    conflict-free selection exists and returns its unique optimum.  When
    it proves that none exists, a complete enumeration over option
    assignments finds the unique best failing one.
    """
    alts, alt_meta, _scale, option_index = _merge_alternatives(options)
    searcher = _RemediationSearch(norm, rules, alts, alt_meta, adjusted,
                                  option_index)
    chosen = searcher.find_release()
    ordered_adjusted = sorted(adjusted)
    if chosen is not None:
        picks = {
            cid: options[cid].index(chosen[cid]) for cid in ordered_adjusted
        }
        result = assess_normalized(
            _modified_norm(norm, ordered_adjusted, options, picks), rules
        )
        total = sum((chosen[c]["cost"] for c in ordered_adjusted), Fraction(0))
        changed = sum(1 for c in ordered_adjusted if chosen[c]["changed"])
        return (total, changed, _choice_sequence(ordered_adjusted, chosen)), \
            chosen, result, None

    failure_best = None
    for combo in product(*[options[cid] for cid in ordered_adjusted]):
        chosen_f = dict(zip(ordered_adjusted, combo))
        picks = {
            cid: options[cid].index(chosen_f[cid]) for cid in ordered_adjusted
        }
        result = assess_normalized(
            _modified_norm(norm, ordered_adjusted, options, picks), rules
        )
        total = sum((opt["cost"] for opt in combo), Fraction(0))
        key = (
            len(result.get("conflicting_components", [])),
            total,
            _choice_sequence(ordered_adjusted, chosen_f),
        )
        if failure_best is None or key < failure_best[0]:
            failure_best = (key, chosen_f, result)
    return None, None, None, failure_best


# ---------------------------------------------------------------------------
# response assembly
# ---------------------------------------------------------------------------

def _conflict_explanation(result):
    """Deterministic human-readable explanation of each remaining conflict."""
    return [
        "%s at [%s]: %s (licenses: %s)"
        % (
            cf["type"],
            ", ".join(cf["components"]),
            cf["detail"],
            ", ".join(cf["licenses"]),
        )
        for cf in result.get("conflicts", [])
    ]


def _build_payload(row, mode, chosen, total_cost, result, candidate_echo,
                   release):
    """Assemble the common response envelope (hash added by the caller)."""
    ordered_adjusted = sorted(chosen)
    originals = {c["id"]: c for c in row["request"]["components"]}

    choices, replacements = [], []
    for cid in ordered_adjusted:
        opt = chosen[cid]
        choices.append(
            {
                "component": cid,
                "candidate_id": opt["candidate_id"],
                "changed": bool(opt["changed"]),
                "cost": _num(opt["cost"]),
            }
        )
        if opt["changed"]:
            replacements.append(
                {
                    "component": cid,
                    "candidate_id": opt["candidate_id"],
                    "from": originals[cid]["license"],
                    "to": opt["canonical"],
                    "cost": _num(opt["cost"]),
                }
            )

    changed = sum(1 for cid in ordered_adjusted if chosen[cid]["changed"])
    payload = {
        "assessment_id": row["id"],
        "release": bool(release),
        "selection_mode": mode,
        "candidates": candidate_echo,
        "choices": choices,
        "replacements": replacements,
        "total_cost": _num(total_cost),
        "changed_components": changed,
        "result": result,
    }
    if release:
        payload["adopted"] = result.get("adopted", {})
        payload["artifact_obligations"] = result.get("artifact_obligations", [])
        payload["components"] = result.get("components", [])
    return payload


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def build_remediation(row, body):
    """Build the remediation plan for a persisted conflict assessment.

    The caller guarantees the assessment exists, carries a verifiable
    rules snapshot and has ``status == "conflict"``.  Raises
    :class:`NoReleaseError` (HTTP 409 payload) when no candidate
    assignment makes the assessment releasable.
    """
    norm = row["request"]
    rules = row["rules"]
    comp_ids = [c["id"] for c in norm["components"]]

    by_component, candidate_echo = normalize_candidates(body, comp_ids, rules)
    options = _build_options(norm, by_component)
    adjusted = set(by_component)

    total_assignments = 1
    for cid in adjusted:
        total_assignments *= len(options[cid])

    if total_assignments <= MAX_OPTION_ASSIGNMENTS:
        mode = "exhaustive"
        release_best, failure_best = _solve_exhaustive(
            norm, rules, adjusted, options
        )
    else:
        mode = "branch-and-bound"
        _key, _chosen, release_result, failure_best = _solve_branch_bound(
            norm, rules, adjusted, options
        )
        if failure_best is None:
            total, changed, _seq = _key
            release_best = (
                (total, changed, _seq), _chosen, release_result
            )
        else:
            release_best = None

    if release_best is not None:
        (total, _changed, _seq), chosen, result = release_best
        payload = _build_payload(
            row, mode, chosen, total, result, candidate_echo, release=True
        )
        payload["remediation_hash"] = stable_hash(payload)
        return payload

    (_n_conf, total, _seq), chosen, result = failure_best
    payload = _build_payload(
        row, mode, chosen, total, result, candidate_echo, release=False
    )
    payload["reason"] = "no-releasable-assignment"
    payload["conflicting_components"] = result.get("conflicting_components", [])
    payload["conflicts"] = result.get("conflicts", [])
    payload["confirmed_subgraph"] = result.get(
        "confirmed_subgraph", {"components": []}
    )
    payload["conflict_explanation"] = _conflict_explanation(result)
    payload["error"] = (
        "no candidate assignment makes assessment %r releasable; the unique "
        "best failing assignment (fewest conflicting components, then lowest "
        "cost) leaves %d conflicting component(s): %s"
        % (
            row["id"],
            len(payload["conflicting_components"]),
            ", ".join(payload["conflicting_components"]),
        )
    )
    payload["remediation_hash"] = stable_hash(payload)
    raise NoReleaseError(payload)

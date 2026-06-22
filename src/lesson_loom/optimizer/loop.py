"""The self-improvement controller.

A plain Python controller around the LangGraph generation graph (the optimizer's
flow is linear with clear decision points, so it reads better as Python than as a
second graph). Greedy, single-axis hill-climbing: each round it proposes one
change, evaluates it, and the promotion gate decides — staying on the best config
when a candidate is rejected. It stops when there are no moves left or after
`patience` consecutive rejects, then reports the best config on the untouched
TEST split (the unbiased headline number).
"""

from __future__ import annotations

from lesson_loom.core.config import default_config
from lesson_loom.core.hashing import content_hash
from lesson_loom.core.logging import get_logger
from lesson_loom.core.schemas import ContextPack, ExperimentResult, ExperimentRound, SystemConfig
from lesson_loom.evals.cache import ResultCache
from lesson_loom.evals.harness import load_eval_items, run_eval
from lesson_loom.optimizer import promotion
from lesson_loom.optimizer.failure_analyzer import analyze
from lesson_loom.optimizer.proposal import ProposalEngine
from lesson_loom.optimizer.store import LineageStore
from lesson_loom.providers.base import LLMProvider

log = get_logger("optimizer")


class OptimizerController:
    def __init__(
        self,
        provider: LLMProvider,
        pack: ContextPack,
        subject: str,
        *,
        strategy: str = "heuristic",
        max_rounds: int = 8,
        patience: int = 3,
        store: LineageStore | None = None,
        baseline: SystemConfig | None = None,
    ):
        self.provider = provider
        self.pack = pack
        self.subject = subject
        self.max_rounds = max_rounds
        self.patience = patience
        self.store = store
        self.engine = ProposalEngine(strategy=strategy, provider=provider)
        self.cache = ResultCache()
        self.baseline = baseline or default_config(provider_name=provider.name)

        self.train = load_eval_items(subject, "train")
        self.heldout = load_eval_items(subject, "heldout")
        self.test = load_eval_items(subject, "test")

    def _eval(self, config: SystemConfig, items, split: str):
        return run_eval(self.provider, config, self.pack, items, split=split, cache=self.cache)

    def run(self) -> ExperimentResult:
        exp_id = content_hash({"subject": self.subject, "baseline": self.baseline.id})
        if self.store:
            self.store.save_config(self.baseline)
            self.store.start_experiment(exp_id, self.subject, self.baseline.id)

        current = self.baseline
        cur_train = self._eval(current, self.train, "train")
        cur_heldout = self._eval(current, self.heldout, "heldout")
        log.info("baseline", northstar=cur_train.northstar, heldout=cur_heldout.northstar)

        rounds: list[ExperimentRound] = []
        attempts: set[tuple[str, str]] = set()
        rejects = 0

        for n in range(1, self.max_rounds + 1):
            gaps = analyze(cur_train, current)
            proposal = self.engine.propose(current, gaps, cur_train, attempts)
            if proposal is None:
                log.info("converged", round=n)
                break

            attempts.add((proposal.axis, proposal.target))
            if self.store:
                self.store.save_config(proposal.config)

            cand_train = self._eval(proposal.config, self.train, "train")
            cand_heldout = self._eval(proposal.config, self.heldout, "heldout")
            decision = promotion.evaluate(
                proposal.axis, proposal.target, cur_train, cur_heldout, cand_train, cand_heldout
            )

            rnd = ExperimentRound(
                round_num=n,
                parent_config_id=current.id,
                candidate_config_id=proposal.config.id,
                axis=proposal.axis,
                target=proposal.target,
                rationale=proposal.rationale,
                mutations=proposal.mutations,
                train_northstar=cand_train.northstar,
                heldout_northstar=cand_heldout.northstar,
                decision=decision,
            )
            rounds.append(rnd)
            if self.store:
                self.store.record_round(exp_id, rnd, cand_train, cand_heldout)

            log.info(
                "round", n=n, axis=proposal.axis, target=proposal.target,
                train=cand_train.northstar, delta=decision.northstar_train_delta,
                promoted=decision.promoted, reason=decision.reason,
            )

            if decision.promoted:
                current, cur_train, cur_heldout = proposal.config, cand_train, cand_heldout
                rejects = 0
            else:
                rejects += 1
                if rejects >= self.patience:
                    log.info("stopping_on_patience", rejects=rejects)
                    break

        baseline_test = self._eval(self.baseline, self.test, "test")
        best_test = self._eval(current, self.test, "test")

        result = ExperimentResult(
            experiment_id=exp_id,
            subject=self.subject,
            rounds=rounds,
            baseline_config_id=self.baseline.id,
            best_config_id=current.id,
            baseline_test_northstar=baseline_test.northstar,
            best_test_northstar=best_test.northstar,
        )
        if self.store:
            self.store.finish_experiment(result)
        return result

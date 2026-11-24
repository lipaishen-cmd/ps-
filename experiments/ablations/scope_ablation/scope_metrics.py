"""Event-level disturbance of old, unfrozen execution subbatches.

New insertions do not count as queue moves. A former UID removed by an allowed
resplit counts as a changed old batch; its new UID has no one-to-one machine or
start-time counterpart and is reported separately, never imputed.
"""
from __future__ import annotations


def _locations(solution):
    result = {}
    for mid, queue in solution.machine_queues.items():
        for pos, batch in enumerate(queue.batches):
            if batch.u_id in result:
                raise ValueError(f"Duplicate batch UID {batch.u_id}")
            result[batch.u_id] = (mid, pos, float(batch.start_time), float(batch.finish_time),
                                  batch.order_id, int(batch.stack_count))
    return result


def event_perturbation(before, after, frozen_uids):
    old = _locations(before)
    new = _locations(after)
    frozen = set(frozen_uids)
    if not frozen <= set(old):
        raise ValueError("Frozen UID absent from event-before solution")
    for uid in frozen:
        if uid not in new or old[uid] != new[uid]:
            raise ValueError(f"Frozen batch changed: {uid}")
    movable = set(old) - frozen
    retained = movable & set(new)
    replaced = movable - set(new)
    migrated = {uid for uid in retained if old[uid][0] != new[uid][0]}
    reordered = set()
    # Compare relative order only for old UIDs retained on the same machine.
    same = sorted(retained - migrated)
    for i, left in enumerate(same):
        for right in same[i + 1:]:
            if old[left][0] != old[right][0] or new[left][0] != new[right][0]:
                continue
            if (old[left][1] < old[right][1]) != (new[left][1] < new[right][1]):
                reordered.update((left, right))
    moved = migrated | reordered | replaced
    shifts = [abs(new[uid][2] - old[uid][2]) for uid in retained]
    return {
        "old_unfrozen_count": len(movable),
        "moved_old_batches": len(moved),
        "cross_machine_migrations": len(migrated),
        "reordered_old_batches": len(reordered),
        "replaced_old_batches": len(replaced),
        "start_shift_observed_batches": len(shifts),
        "start_shift_abs_sum_min": sum(shifts),
        "mean_start_shift_min": sum(shifts) / len(shifts) if shifts else None,
    }

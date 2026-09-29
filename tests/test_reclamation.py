"""End-to-end tests for historical revision reclamation
(``POST /revision-reclamation``).

Covers: request validation and server-side de-duplication, 404 when any
path (or empty placeholder) does not exist, the strictly-older-than
``before`` cutoff, retention of the current revision and of revisions
pinned by copy-session manifests or target baselines, dry-run/executed
consistency, idempotent repeats with monotonic revision numbers, whole
batch rollback, and concurrency against PUT, streaming GET and
session creation/acknowledgement.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import threading
import time
import uuid
from datetime import datetime, timezone

import asyncpg

SHA_OF = lambda body: hashlib.sha256(body).hexdigest()

RECLAIM = "/revision-reclamation"
FUTURE = "2030-01-01T00:00:00Z"
FUTURE_DT = datetime(2030, 1, 1, tzinfo=timezone.utc)
# A fixed "old" instant for revisions that must fall in the reclaim
# window, and a cutoff that includes them but not revisions created now.
OLD = datetime(2026, 1, 15, tzinfo=timezone.utc)
BEFORE = "2026-02-01T00:00:00Z"


def _p(name: str) -> str:
    return f"rc/{name}-{uuid.uuid4().hex[:12]}.txt"


def _reclaim(client, paths, before=FUTURE, dry_run=True):
    return client.post(
        RECLAIM, json={"paths": paths, "before": before, "dry_run": dry_run}
    )


def _publish_revisions(client, path, bodies):
    for body in bodies:
        r = client.put(f"/objects/{path}", content=body)
        assert r.status_code == 204


async def _backdate(path, revision_no, created_at=OLD):
    """Pin a revision's creation time (created_at defaults to now())."""
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        await conn.execute(
            "UPDATE revisions SET created_at = $2"
            " WHERE object_path = $1 AND revision_no = $3",
            path,
            created_at,
            revision_no,
        )
    finally:
        await conn.close()


def _result_for(data, path):
    return next(r for r in data["paths"] if r["path"] == path)


def _nos(data, path):
    return [c["revision_no"] for c in _result_for(data, path)["candidates"]]


def _retained(data, path):
    return {r["revision_no"]: r["reason"] for r in _result_for(data, path)["retained"]}


def _create_session_for(client, src, prefix, segment_size=1):
    return client.post(
        "/copy-sessions",
        json={
            "source_paths": src,
            "target_prefix": prefix,
            "segment_size": segment_size,
        },
    ).json()["session"]


# ---- request validation -------------------------------------------------------

def test_validation_errors(client):
    assert client.post(RECLAIM, json={}).status_code == 400
    assert client.post(RECLAIM, json={"paths": []}).status_code == 400
    assert (
        client.post(
            RECLAIM,
            json={"paths": [_p("x")], "before": FUTURE},
        ).status_code
        == 400
    )
    assert (
        client.post(
            RECLAIM,
            json={"paths": [_p("x")], "dry_run": True},
        ).status_code
        == 400
    )
    assert (
        client.post(
            RECLAIM,
            json={"paths": [_p("x")], "before": FUTURE, "dry_run": "yes"},
        ).status_code
        == 400
    )
    assert (
        client.post(
            RECLAIM,
            json={"paths": ["a", ""], "before": FUTURE, "dry_run": True},
        ).status_code
        == 400
    )
    assert (
        client.post(
            RECLAIM,
            json={"paths": [1, 2], "before": FUTURE, "dry_run": True},
        ).status_code
        == 400
    )
    # Naive (timezone-less) timestamps are refused, not assumed UTC.
    assert (
        client.post(
            RECLAIM,
            json={"paths": ["a"], "before": "2030-01-01T00:00:00", "dry_run": True},
        ).status_code
        == 400
    )
    assert (
        client.post(
            RECLAIM,
            json={"paths": ["a"], "before": "not-a-date", "dry_run": True},
        ).status_code
        == 400
    )
    assert client.post(RECLAIM, content=b"not json").status_code == 400


def test_too_many_paths_is_400(client):
    paths = [f"rc/many/{i}.txt" for i in range(101)]
    assert _reclaim(client, paths).status_code == 400


def test_up_to_100_paths_accepted(client):
    paths = [_p(f"many-{i}") for i in range(100)]
    for p in paths:
        client.put(f"/objects/{p}", content=b"x")
    r = _reclaim(client, paths)
    assert r.status_code == 200
    assert [x["path"] for x in r.json()["paths"]] == sorted(paths)


# ---- 404 semantics -------------------------------------------------------------

def test_missing_path_whole_request_is_404(client):
    p = _p("exists")
    client.put(f"/objects/{p}", content=b"v1")
    client.put(f"/objects/{p}", content=b"v2")
    missing = _p("nope")
    r = _reclaim(client, [p, missing])
    assert r.status_code == 404
    assert r.json() == {"detail": "path_not_found", "missing": [missing]}
    # Nothing was reclaimed on the existing path either.
    assert client.get(f"/objects/{p}").content == b"v2"


def test_empty_placeholder_path_is_404(client):
    # A current_id-NULL placeholder row (as left by copy-session creation
    # for a nonexistent target) is "not found" for reclamation purposes.
    placeholder = _p("placeholder")

    async def make_placeholder():
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            await conn.execute(
                "INSERT INTO objects (path) VALUES ($1)"
                " ON CONFLICT (path) DO NOTHING",
                placeholder,
            )
        finally:
            await conn.close()

    asyncio.run(make_placeholder())
    r = _reclaim(client, [placeholder], dry_run=False)
    assert r.status_code == 404
    assert r.json()["missing"] == [placeholder]
    # The placeholder row itself is left intact.
    async def current_id():
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            return await conn.fetchval(
                "SELECT current_id FROM objects WHERE path = $1", placeholder
            )
        finally:
            await conn.close()

    assert asyncio.run(current_id()) is None


# ---- dry run: cutoff, current, dedup, ordering ----------------------------------

def test_dry_run_lists_only_old_non_current_revisions(client):
    p = _p("basic")
    bodies = [b"rev-1", b"rev-2-longer", b"rev-3-current"]
    _publish_revisions(client, p, bodies)
    asyncio.run(_backdate(p, 1))
    asyncio.run(_backdate(p, 2))

    r = _reclaim(client, [p])
    assert r.status_code == 200
    data = r.json()
    assert data["dry_run"] is True
    assert data["before"] == "2030-01-01T00:00:00+00:00"
    res = _result_for(data, p)
    assert _nos(data, p) == [1, 2]
    assert res["reclaimable_bytes"] == len(bodies[0]) + len(bodies[1])
    assert data["total_reclaimable_bytes"] == res["reclaimable_bytes"]
    assert data["total_candidates"] == 2
    assert _retained(data, p) == {3: "current"}
    # Candidate entries carry number, bytes, digest and creation time.
    assert res["candidates"][0] == {
        "revision_no": 1,
        "sha256": SHA_OF(b"rev-1"),
        "bytes": 5,
        "created_at": OLD.isoformat(),
    }
    # Current content is untouched by a dry run.
    assert client.get(f"/objects/{p}").content == bodies[2]


def test_cutoff_is_strict_before(client):
    p = _p("cutoff")
    _publish_revisions(client, p, [b"a", b"b"])
    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    asyncio.run(_backdate(p, 1, at))
    asyncio.run(_backdate(p, 2, at))
    # before == created_at excludes the revision (strictly older only).
    r = _reclaim(client, [p], before="2026-01-01T00:00:00+00:00")
    assert _nos(r.json(), p) == []
    assert _retained(r.json(), p) == {}  # neither revision is considered
    # One second later: rev 1 is a candidate, rev 2 (current) is retained.
    r2 = _reclaim(client, [p], before="2026-01-01T00:00:01Z")
    assert _nos(r2.json(), p) == [1]
    assert _retained(r2.json(), p) == {2: "current"}
    # One second earlier than both: nothing considered.
    r3 = _reclaim(client, [p], before="2025-12-31T23:59:59Z")
    assert _nos(r3.json(), p) == []


def test_paths_are_deduplicated_and_sorted(client):
    p1, p2 = _p("a"), _p("b")
    _publish_revisions(client, p1, [b"1", b"2"])
    _publish_revisions(client, p2, [b"1", b"2"])
    r = _reclaim(client, [p2, p1, p2, p1])  # duplicates, unsorted
    assert r.status_code == 200
    assert [x["path"] for x in r.json()["paths"]] == [p1, p2]
    assert r.json()["total_candidates"] == 2  # one old revision per path


def test_recent_revisions_after_cutoff_are_not_considered(client):
    # A revision newer than `before` is neither a candidate nor retained.
    p = _p("recent")
    client.put(f"/objects/{p}", content=b"only-current")
    r = _reclaim(client, [p], before="2020-01-01T00:00:00Z")
    res = _result_for(r.json(), p)
    assert res["candidates"] == []
    assert res["retained"] == []
    assert res["reclaimable_bytes"] == 0


# ---- retention: copy-session manifest and target baseline ------------------------

def test_manifest_pinned_revision_is_retained(client):
    src = _p("src")
    # Create the session while rev 1 is current: the manifest pins rev 1.
    client.put(f"/objects/{src}", content=b"snap")
    asyncio.run(_backdate(src, 1))
    sid = _create_session_for(client, [src], "bak/")
    # Publish a newer revision afterwards; the pinned snapshot must not move.
    client.put(f"/objects/{src}", content=b"new-current")

    r = _reclaim(client, [src])
    assert _retained(r.json(), src) == {
        1: "copy_session_manifest",
        2: "current",
    }
    assert _nos(r.json(), src) == []

    # Even after the session is fully applied, its manifest row still
    # exists, so the pinned revision stays retained.
    cursor = client.get(f"/copy-sessions/{sid}/segments").json()["cursor"]
    assert client.post(
        f"/copy-sessions/{sid}/acks", json={"cursor": cursor}
    ).status_code == 200
    r2 = _reclaim(client, [src])
    assert _retained(r2.json(), src) == {
        1: "copy_session_manifest",
        2: "current",
    }


def test_baseline_pinned_revision_is_retained(client):
    src = _p("bsrc")
    client.put(f"/objects/{src}", content=b"payload")
    target = "bak2/" + src
    # Pin the session while rev 2 of the target is current: the baseline
    # references rev 2; rev 1 has no reference.
    client.put(f"/objects/{target}", content=b"old-target")
    client.put(f"/objects/{target}", content=b"payload")
    asyncio.run(_backdate(target, 1))
    asyncio.run(_backdate(target, 2))
    _create_session_for(client, [src], "bak2/")
    # Then publish rev 3 at the target (still inside the far-future
    # cutoff) so the baseline revision is no longer current.
    client.put(f"/objects/{target}", content=b"fresh-current")

    r = _reclaim(client, [target])
    assert _nos(r.json(), target) == [1]
    assert _retained(r.json(), target) == {
        2: "target_baseline",
        3: "current",
    }


def test_current_takes_priority_over_baseline_reason(client):
    src = _p("prio-src")
    client.put(f"/objects/{src}", content=b"payload")
    target = "bak3/" + src
    client.put(f"/objects/{target}", content=b"payload")
    asyncio.run(_backdate(target, 1))
    _create_session_for(client, [src], "bak3/")  # baseline pins rev 1 == current
    r = _reclaim(client, [target])
    assert _retained(r.json(), target) == {1: "current"}


# ---- execution -----------------------------------------------------------------

def test_execute_deletes_candidates_and_is_idempotent(client):
    p = _p("exec")
    _publish_revisions(client, p, [b"v1", b"v2-long", b"v3"])
    asyncio.run(_backdate(p, 1))
    asyncio.run(_backdate(p, 2))

    preview = _reclaim(client, [p], dry_run=True).json()
    assert _nos(preview, p) == [1, 2]

    r = _reclaim(client, [p], dry_run=False)
    assert r.status_code == 200
    data = r.json()
    assert data["dry_run"] is False
    assert data["deleted_revisions"] == 2
    assert data["reclaimed_bytes"] == len(b"v1") + len(b"v2-long")
    assert data["deleted"] == [
        {
            "path": p,
            "deleted": [
                {
                    "revision_no": 1,
                    "bytes": len(b"v1"),
                    "created_at": OLD.isoformat(),
                },
                {
                    "revision_no": 2,
                    "bytes": len(b"v2-long"),
                    "created_at": OLD.isoformat(),
                },
            ],
        }
    ]

    # Current revision is intact and readable.
    assert client.get(f"/objects/{p}").content == b"v3"

    # Repeat: per-path empty lists, zero bytes; later publishes keep increasing.
    again = _reclaim(client, [p], dry_run=False)
    assert again.status_code == 200
    assert again.json()["deleted"] == [{"path": p, "deleted": []}]
    assert again.json()["deleted_revisions"] == 0
    assert again.json()["reclaimed_bytes"] == 0

    next_put = client.put(f"/objects/{p}", content=b"v4")
    assert next_put.status_code == 204
    assert next_put.headers["x-revision-no"] == "4"  # deleted numbers never reused
    assert client.get(f"/objects/{p}").content == b"v4"


def test_dry_run_does_not_mutate(client):
    p = _p("dry")
    _publish_revisions(client, p, [b"a", b"b", b"c"])
    asyncio.run(_backdate(p, 1))
    asyncio.run(_backdate(p, 2))
    r1 = _reclaim(client, [p], dry_run=True)
    r2 = _reclaim(client, [p], dry_run=True)
    assert r1.content == r2.content
    assert _nos(r1.json(), p) == [1, 2]


def test_execute_with_recent_cutoff_then_republish_continues_numbers(client):
    # A recent cutoff deletes only the backdated non-current revision;
    # the just-published revision is outside the window and survives.
    p = _p("rejudge")
    client.put(f"/objects/{p}", content=b"first")
    asyncio.run(_backdate(p, 1))
    client.put(f"/objects/{p}", content=b"second")
    # Rev 1 is old AND non-current -> candidate; rev 2 is recent.
    assert _nos(_reclaim(client, [p], before=BEFORE).json(), p) == [1]
    r = _reclaim(client, [p], before=BEFORE, dry_run=False)
    assert r.status_code == 200
    assert r.json()["deleted_revisions"] == 1
    assert client.get(f"/objects/{p}").content == b"second"
    # The next publish continues the sequence across the deleted gap.
    assert client.put(f"/objects/{p}", content=b"third").headers["x-revision-no"] == "3"


def test_execute_missing_path_is_404_and_changes_nothing(client):
    p = _p("keep")
    _publish_revisions(client, p, [b"a", b"b"])
    asyncio.run(_backdate(p, 1))
    r = _reclaim(client, [p, _p("ghost")], before=BEFORE, dry_run=False)
    assert r.status_code == 404
    # The existing path's old revision must survive the aborted batch.
    preview = _reclaim(client, [p], before=BEFORE).json()
    assert _nos(preview, p) == [1]
    assert client.get(f"/objects/{p}").content == b"b"


# ---- concurrency ----------------------------------------------------------------

def test_reclaim_vs_concurrent_publish_never_destroys_in_use(client):
    """Barrier race over HTTP, repeated: execution (FOR UPDATE) vs PUT.

    Whichever transaction takes the object-row lock first wins; the loser
    wakes to the committed state.  The reclaim must never delete a
    revision the PUT makes current, and revision numbers stay monotonic.
    """
    for i in range(8):
        p = _p(f"race-put-{i}")
        _publish_revisions(client, p, [b"old-1", b"old-2", b"current-3"])
        asyncio.run(_backdate(p, 1))
        asyncio.run(_backdate(p, 2))

        barrier = threading.Barrier(2)
        out = {}

        def reclaim():
            barrier.wait()
            out["rec"] = _reclaim(client, [p], before=BEFORE, dry_run=False)

        def publish():
            barrier.wait()
            out["put"] = client.put(f"/objects/{p}", content=b"concurrent-4")

        threads = [threading.Thread(target=reclaim), threading.Thread(target=publish)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        rec, put = out["rec"], out["put"]
        assert rec.status_code == 200
        assert put.status_code == 204
        # Exactly the two backdated revisions are reclaimed; rev 3 (which
        # was current when the race began) is never in the cutoff window.
        assert rec.json()["deleted_revisions"] == 2
        deleted_nos = [
            e["revision_no"]
            for item in rec.json()["deleted"]
            for e in item["deleted"]
        ]
        assert sorted(deleted_nos) == [1, 2]
        # The PUT always lands revision 4 (3 exists; 1/2 gaps are not reused).
        assert put.headers["x-revision-no"] == "4"
        g = client.get(f"/objects/{p}")
        assert g.content == b"concurrent-4"
        # Repeated reclaim after the race is a per-path empty no-op.
        again = _reclaim(client, [p], before=BEFORE, dry_run=False).json()
        assert again["deleted"] == [{"path": p, "deleted": []}]


def test_reclaim_blocks_behind_streaming_get(client):
    """A streaming GET holds the object FOR SHARE until the last byte; an
    execution waits for it and then never removes the streamed revision.

    A 16 MiB body with a slow client read fills the loopback socket
    buffers, so the server-side generator (and its transaction) is still
    open -- blocked on backpressure -- when the reclaim reaches its
    FOR UPDATE wait.
    """
    p = _p("race-stream")
    v1 = b"A" * (16 * 1024 * 1024)
    v2 = b"B" * (16 * 1024 * 1024)
    client.put(f"/objects/{p}", content=v1)
    etag2 = client.put(f"/objects/{p}", content=v2).headers["etag"]
    asyncio.run(_backdate(p, 1))

    done = {}

    def reclaim():
        done["rec"] = _reclaim(client, [p], before=BEFORE, dry_run=False)

    with client.stream("GET", f"/objects/{p}") as resp:
        assert resp.status_code == 200
        assert resp.headers["etag"] == etag2
        it = resp.iter_bytes()
        first = next(it)
        t = threading.Thread(target=reclaim)
        t.start()
        time.sleep(1.0)  # server generator blocks on socket backpressure
        assert t.is_alive()  # queued behind the in-flight GET's FOR SHARE
        # Drain slowly so the stream stays open until the assertion.
        rest = bytearray()
        for chunk in it:
            rest.extend(chunk)
    t.join(timeout=30)
    assert not t.is_alive()
    body = first + bytes(rest)
    assert body == v2  # streamed revision was never removed
    assert done["rec"].status_code == 200
    assert done["rec"].json()["deleted_revisions"] == 1  # only backdated rev 1
    assert client.get(f"/objects/{p}").content == v2


def test_reclaim_waits_for_held_for_share_lock_db_level(client):
    """Deterministic equivalent of the streaming-GET interleave: a read
    transaction holding FOR SHARE (exactly what GET holds until the last
    byte) makes the reclaim's FOR UPDATE wait; once the read commits the
    reclaim proceeds and keeps the current revision."""
    from app import db

    p = _p("db-stream")
    client.put(f"/objects/{p}", content=b"old")
    asyncio.run(_backdate(p, 1))
    client.put(f"/objects/{p}", content=b"current")

    async def scenario():
        conn_get = await asyncpg.connect(os.environ["DATABASE_URL"])
        conn_rec = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            # GET: open transaction, take FOR SHARE, keep it open like a
            # response still streaming bytes.
            t_get = conn_get.transaction()
            await t_get.start()
            rev = await db.current_revision(conn_get, p, for_update=False)
            assert rev is not None and rev.revision_no == 2

            async def reclaim():
                tx = conn_rec.transaction()
                await tx.start()
                current = await db.lock_reclaim_paths(conn_rec, [p])
                seen = await db.collect_reclaimable(conn_rec, [p], FUTURE_DT)
                to_delete = [r["id"] for r in seen[p] if db.reclaim_reason(r) is None]
                deleted = await db.delete_revisions(conn_rec, to_delete)
                await tx.commit()
                return deleted

            task = asyncio.create_task(reclaim())
            await asyncio.sleep(1.0)
            assert not task.done()  # FOR UPDATE queued behind the FOR SHARE
            await t_get.commit()  # stream finished: lock released
            deleted = await asyncio.wait_for(task, timeout=10)
            assert [r["revision_no"] for r in deleted] == [1]
        finally:
            await conn_get.close()
            await conn_rec.close()

    asyncio.run(scenario())
    assert client.get(f"/objects/{p}").content == b"current"


def test_reclaim_vs_session_creation_leaves_no_dangling_reference(client):
    """Reclaim racing copy-session creation: whichever wins, the manifest
    snapshot that the session serves must remain readable and applicable.

    The cutoff includes both revisions, so if the session takes the lock
    first its manifest pins the current rev 2 (reported retained); if the
    reclaim commits first, rev 1 (non-current, unreferenced) is removed
    and the session then snapshots rev 2 -- either way the served
    manifest is intact.
    """
    for i in range(6):
        src = _p(f"race-src-{i}")
        client.put(f"/objects/{src}", content=b"snap-old")
        client.put(f"/objects/{src}", content=b"snap-current")
        asyncio.run(_backdate(src, 1))

        barrier = threading.Barrier(2)
        out = {}

        def reclaim():
            barrier.wait()
            out["rec"] = _reclaim(client, [src], dry_run=False)

        def create():
            barrier.wait()
            out["cs"] = client.post(
                "/copy-sessions",
                json={"source_paths": [src], "target_prefix": f"racepre/{i}/"},
            )

        threads = [threading.Thread(target=reclaim), threading.Thread(target=create)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert out["rec"].status_code == 200
        assert out["cs"].status_code == 201
        sid = out["cs"].json()["session"]

        # The manifest is fully servable: pinned content reads back whole.
        seg = client.get(f"/copy-sessions/{sid}/segments")
        assert seg.status_code == 200
        entries = seg.json()["entries"]
        assert len(entries) == 1
        assert base64.b64decode(entries[0]["content_b64"]) == b"snap-current"
        pinned_no = entries[0]["revision_no"]

        # Whatever the manifest pins must be a retained revision now.
        after = _reclaim(client, [src]).json()
        assert pinned_no in _retained(after, src)
        assert pinned_no not in _nos(after, src)

        # Acknowledgement applies without hitting a dangling reference.
        ack = client.post(
            f"/copy-sessions/{sid}/acks",
            json={"cursor": seg.json()["cursor"]},
        )
        assert ack.status_code in (200, 409)
        assert client.get(f"/objects/{src}").status_code == 200


def test_reclaim_vs_session_ack_serializes_without_dangling_reference(client):
    """Reclaim racing a segment acknowledgement: both take FOR UPDATE in
    ascending path order, so they serialise.  The baseline revision is
    retained and the ack's verdict stays valid after the reclaim."""
    src = _p("ack-src")
    client.put(f"/objects/{src}", content=b"payload")
    target = "raceackpre/" + src
    client.put(f"/objects/{target}", content=b"old-body")
    client.put(f"/objects/{target}", content=b"payload")  # == session content
    asyncio.run(_backdate(target, 1))
    asyncio.run(_backdate(target, 2))
    sid = _create_session_for(client, [src], "raceackpre/")
    cursor = client.get(f"/copy-sessions/{sid}/segments").json()["cursor"]

    barrier = threading.Barrier(2)
    out = {}

    def reclaim():
        barrier.wait()
        out["rec"] = _reclaim(client, [target], dry_run=False)

    def ack():
        barrier.wait()
        out["ack"] = client.post(
            f"/copy-sessions/{sid}/acks", json={"cursor": cursor}
        )

    threads = [threading.Thread(target=reclaim), threading.Thread(target=ack)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert out["rec"].status_code == 200
    assert out["ack"].status_code == 200  # already_present, never a 500
    # Only the unreferenced old revision was reclaimed.
    assert out["rec"].json()["deleted_revisions"] == 1
    assert client.get(f"/objects/{target}").content == b"payload"
    # Nothing left to reclaim; rev 2 survives -- it is current and also
    # the pinned baseline (the current reason wins by priority).
    after = _reclaim(client, [target]).json()
    assert _nos(after, target) == []
    assert 2 in _retained(after, target)
    assert client.get(f"/copy-sessions/{sid}").json()["status"] == "complete"


# ---- database-level interleave / integrity ----------------------------------------

def test_rejudge_under_lock_tracks_committed_publish(client):
    """DB-level: an execution holding FOR UPDATE re-judges inside the
    lock; a PUT that commits first is visible as the new current and its
    revision is never deleted."""
    from app import db

    p = _p("db-race")
    client.put(f"/objects/{p}", content=b"one")
    asyncio.run(_backdate(p, 1))

    async def scenario():
        conn_put = await asyncpg.connect(os.environ["DATABASE_URL"])
        conn_rec = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            # Reclaim locks the object row first.
            t_rec = conn_rec.transaction()
            await t_rec.start()
            current = await db.lock_reclaim_paths(conn_rec, [p])
            assert current[p] is not None

            # The PUT arrives and blocks on the row lock.
            async def put():
                t_put = conn_put.transaction()
                await t_put.start()
                await db.lock_new_or_existing_object(conn_put, p)
                _, no, _ = await db.publish_revision(
                    conn_put, p, b"two", "application/octet-stream"
                )
                await t_put.commit()
                return no

            put_task = asyncio.create_task(put())
            await asyncio.sleep(1.0)
            assert not put_task.done()  # queued, not deadlocked

            # Snapshot inside the held lock: the uncommitted PUT is
            # invisible; only rev 1 is there.
            seen = await db.collect_reclaimable(conn_rec, [p], FUTURE_DT)
            assert [r["revision_no"] for r in seen[p]] == [1]

            # Release without deleting; the PUT proceeds as revision 2.
            await t_rec.commit()
            assert await asyncio.wait_for(put_task, timeout=10) == 2

            # Real reclaim now: rev 1 is unreferenced -> deleted; rev 2 is
            # current -> kept.
            t2 = conn_rec.transaction()
            await t2.start()
            current2 = await db.lock_reclaim_paths(conn_rec, [p])
            seen2 = await db.collect_reclaimable(conn_rec, [p], FUTURE_DT)
            to_delete = [r["id"] for r in seen2[p] if db.reclaim_reason(r) is None]
            deleted = await db.delete_revisions(conn_rec, to_delete)
            await t2.commit()
            assert [r["revision_no"] for r in deleted] == [1]
        finally:
            await conn_put.close()
            await conn_rec.close()

    asyncio.run(scenario())
    assert client.get(f"/objects/{p}").content == b"two"
    # Revision numbers are not reused after the delete.
    assert client.put(f"/objects/{p}", content=b"three").headers["x-revision-no"] == "3"


def test_foreign_key_blocks_deleting_referenced_revision(client):
    """Defence in depth: a direct delete of a manifest-pinned revision
    violates the copy_session_entries foreign key, so the surrounding
    transaction must roll back rather than strand a reference."""
    src = _p("fk-src")
    client.put(f"/objects/{src}", content=b"pinned")
    asyncio.run(_backdate(src, 1))
    _create_session_for(client, [src], "fkpre/")

    async def scenario():
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            rev_id = await conn.fetchval(
                "SELECT id FROM revisions WHERE object_path = $1 AND revision_no = 1",
                src,
            )
            tx = conn.transaction()
            await tx.start()
            try:
                await conn.execute("DELETE FROM revisions WHERE id = $1", rev_id)
            except asyncpg.ForeignKeyViolationError:
                await tx.rollback()
                return "blocked"
            await tx.rollback()
            return "unexpected"
        finally:
            await conn.close()

    assert asyncio.run(scenario()) == "blocked"
    # The session still serves the pinned bytes end to end.
    sid = _create_session_for(client, [src], "fkpre2/")
    seg = client.get(f"/copy-sessions/{sid}/segments").json()
    assert base64.b64decode(seg["entries"][0]["content_b64"]) == b"pinned"

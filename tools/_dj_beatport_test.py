"""Gate for the Beatport integration (lib/dj/beatport.py).

Everything runs against a FAKE transport returning canned Beatport JSON -
no network, no credentials. Covers: token store + expiry + refresh, the
authenticated client (search / track / 401-retry), JSON->row normalization
across field shapes, Camelot conversion (explicit fields AND name parsing),
ghost TrackInfo + tempo/key fit vs a track and vs a library, preview/track
URL derivation, and the wishlist round-trip.

Usage: python tools/_dj_beatport_test.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


def bp_track(tid, name, artist, bpm, key, uuid=None, slug=None):
    return {"id": tid, "name": name,
            "artists": [{"name": artist}],
            "bpm": bpm, "key": key, "length_ms": 380000,
            "slug": slug or name.lower().replace(" ", "-"),
            "sample_uuid": uuid or f"uuid-{tid}",
            "price": {"display": "$1.49"}, "genre": {"name": "Melodic House"},
            "isrc": f"ISRC{tid}"}


def main():
    from lib.dj import beatport as BP

    print("Beatport integration test\n" + "=" * 40 + "\n")

    # -- token store + expiry --------------------------------------------------
    tmp = tempfile.mkdtemp(prefix="gl_bp_")
    tok_path = os.path.join(tmp, "tok.json")
    auth = BP.BeatportAuth(token_path=tok_path, client_id="cid")
    check("starts unauthenticated", not auth.is_authenticated(),
          f"auth={auth.is_authenticated()}")
    auth.set_token_json('{"access_token": "AAA", "refresh_token": "RRR", '
                        '"expires_in": 3600, "token_type": "bearer"}')
    check("accepts pasted token json", auth.is_authenticated()
          and auth.bearer() == "Bearer AAA", f"bearer={auth.bearer()}")
    auth2 = BP.BeatportAuth(token_path=tok_path, client_id="cid")
    check("token persists to disk", auth2.is_authenticated(),
          f"reloaded={auth2.token.get('access_token') if auth2.token else None}")
    # Force expiry and refresh via a fake transport.
    auth.token["obtained_at"] = 0
    check("detects expiry", auth.is_expired(), f"exp_at={auth.expires_at()}")

    def refresh_transport(method, url, headers=None, params=None, data=None):
        if data and data.get("grant_type") == "refresh_token":
            return 200, {"access_token": "BBB", "expires_in": 3600,
                         "token_type": "bearer"}
        return 400, {"detail": "unexpected"}
    ok = auth.refresh(refresh_transport)
    check("refresh swaps the access token", ok and auth.token["access_token"]
          == "BBB" and auth.token["refresh_token"] == "RRR",
          f"token={auth.token['access_token']}")

    # -- username/password login flow (fake session, no network) --------------
    class FakeSession:
        def __init__(self):
            self.posts = []

        def get_text(self, url):
            if url.endswith("docs/"):
                return '<script src="/static/app.js"></script>'
            return "var x = { API_CLIENT_ID: 'scraped_cid_123' };"

        def post_json(self, url, obj):
            self.posts.append(("login", obj))
            return {"username": "dj_dave", "email": "d@e.com"}

        def get_location(self, url):
            return ("https://api.beatport.com/v4/auth/o/post-message/"
                    "?code=AUTHCODE99"), ""

        def post_form(self, url):
            self.posts.append(("token", url))
            return {"access_token": "LOGGED_IN_TOK", "expires_in": 3600,
                    "refresh_token": "RT", "token_type": "bearer"}

    sess = FakeSession()
    la = BP.BeatportAuth(token_path=os.path.join(tmp, "login.json"))
    acct = BP.password_login(la, "dj_dave", "hunter2", session=sess)
    check("scrapes client_id from docs", la.client_id == "scraped_cid_123",
          f"cid={la.client_id}")
    check("password login stores the token", la.is_authenticated()
          and la.token["access_token"] == "LOGGED_IN_TOK"
          and acct["username"] == "dj_dave",
          f"authed={la.is_authenticated()}")
    # The password must NOT be persisted anywhere in the token file.
    import json as _json
    saved = _json.dumps(_json.load(open(la.token_path)))
    check("password is never stored", "hunter2" not in saved,
          "token file has no password")
    # A login rejected by Beatport surfaces the error, no token stored.
    class RejectSession(FakeSession):
        def post_json(self, url, obj):
            return {"error": "invalid credentials"}
    bad = BP.BeatportAuth(token_path=os.path.join(tmp, "bad.json"))
    try:
        BP.password_login(bad, "x", "y", session=RejectSession())
        rejected = False
    except BP.BeatportError:
        rejected = True
    check("bad credentials raise, no token", rejected
          and not bad.is_authenticated(), f"rejected={rejected}")

    # -- client: search / track / 401-retry ------------------------------------
    calls = {"n": 0}

    def fake_api(method, url, headers=None, params=None, data=None):
        calls["n"] += 1
        if data and data.get("grant_type") == "refresh_token":
            return 200, {"access_token": "CCC", "expires_in": 3600}
        auth_hdr = (headers or {}).get("Authorization", "")
        if "catalog/search" in url:
            # First call with the stale token 401s; after refresh it works.
            if auth_hdr == "Bearer STALE":
                return 401, {"detail": "expired"}
            return 200, {"tracks": [
                bp_track(1, "Aurora", "Kllo", 122, KEY_8A),
                bp_track(2, "Deep End", "Lane 8", 124, KEY_9A)]}
        if "catalog/tracks/" in url:
            return 200, bp_track(3, "Solo", "Yotto", 123, KEY_NAME)
        return 404, {"detail": url}

    KEY_8A = {"camelot_number": 8, "camelot_letter": "A", "name": "A min"}
    KEY_9A = {"camelot_number": 9, "camelot_letter": "A", "name": "E min"}
    KEY_NAME = {"name": "F# min"}          # no camelot fields -> parse name

    cauth = BP.BeatportAuth(token_path=os.path.join(tmp, "c.json"),
                            client_id="cid")
    cauth.set_token_json({"access_token": "STALE", "refresh_token": "R",
                          "expires_in": 3600, "token_type": "bearer"})
    client = BP.BeatportClient(auth=cauth, transport=fake_api)
    results = client.search("melodic", per_page=2)
    check("search returns tracks (after 401 refresh)", len(results) == 2
          and results[0]["name"] == "Aurora"
          and cauth.token["access_token"] == "CCC",
          f"{len(results)} results, token now {cauth.token['access_token']}")
    trk = client.track(3)
    check("track fetch", trk["name"] == "Solo", f"got {trk.get('name')}")

    # -- normalization + camelot -----------------------------------------------
    row = BP.beatport_row(results[0])
    check("row normalized", row["id"] == "bp:1" and row["bpm"] == 122.0
          and row["artist"] == "Kllo" and abs(row["duration_s"] - 380) < 1,
          f"row={{'id': {row['id']}, 'bpm': {row['bpm']}, "
          f"'artist': {row['artist']}}}")
    check("camelot from explicit fields", row["camelot"] == "8A",
          f"camelot={row['camelot']}")
    row_named = BP.beatport_row(bp_track(9, "X", "Y", 120, KEY_NAME))
    check("camelot parsed from key name", row_named["camelot"] != "",
          f"F# min -> {row_named['camelot']}")
    check("preview + track urls", row["preview"].endswith("uuid-1.LOFI.mp3")
          and "beatport.com/track/" in (row["url"] or ""),
          f"preview={row['preview']}")

    # -- ghost TrackInfo + fit -------------------------------------------------
    ghost = BP.ghost_trackinfo(row)
    check("ghost TrackInfo scores", ghost.bpm == 122.0
          and ghost.camelot == "8A" and 0.0 <= ghost.energy_proxy() <= 1.0,
          f"bpm={ghost.bpm} camelot={ghost.camelot}")
    # Fit vs a compatible current track (124 bpm, 9A neighbour).
    cur = BP.ghost_trackinfo(BP.beatport_row(results[1]))   # 124, 9A
    fit = BP.fit_vs_track(cur, ghost)
    check("fit vs track: mixable neighbour", fit["mixable"]
          and abs(fit["stretch_pct"]) < 6 and fit["key_fit"] >= 0.9,
          f"fit={fit}")
    far = BP.ghost_trackinfo(BP.beatport_row(
        bp_track(7, "Slow", "Z", 92, KEY_8A)))   # 92 vs 124 -> no stretch
    fit_far = BP.fit_vs_track(cur, far)
    check("fit vs track: tempo gap = fade only", not fit_far["mixable"],
          f"verdict={fit_far['verdict']}")

    # Fit vs a small library.
    from lib.dj.brain import TrackInfo

    def libt(tid, bpm, cam):
        return TrackInfo({"id": tid, "path": str(tid), "title": f"t{tid}",
                          "artist": "a", "duration_s": 360, "bpm": bpm,
                          "bpm_conf": 1.0, "downbeat_offset": 0,
                          "downbeat_conf": 1.0, "camelot": cam,
                          "beat_grid": [], "loudness_gain_db": 0.0,
                          "kick_offset_s": 0.0, "phrase_beats": 0,
                          "phrase_start_s": 0.0, "phrase_conf": 0.0,
                          "mood_hist": {}, "rhythm_density": 0.0,
                          "spectral": {}, "axes": {}, "auto_tags": [],
                          "content_hash": str(tid), "energy_curve": []},
                         [], [], [])
    lib = [libt(1, 123, "8A"), libt(2, 125, "9A"), libt(3, 150, "3B")]
    libfit = BP.fit_vs_library(lib, ghost)   # 122, 8A
    check("fit vs library counts neighbours",
          libfit["mixable_neighbours"] == 2 and libfit["best"][0]["key_fit"]
          >= 0.9, f"neighbours={libfit['mixable_neighbours']} "
          f"best={libfit['best'][:1]}")

    # -- wishlist round-trip ---------------------------------------------------
    wl = BP.Wishlist(tmp)
    added = wl.add(row)
    dup = wl.add(row)
    check("wishlist add + dedup", added and not dup and len(wl.items) == 1,
          f"items={len(wl.items)}")
    wl2 = BP.Wishlist(tmp)          # reload from disk
    check("wishlist persists", len(wl2.items) == 1
          and wl2.items[0]["bp_id"] == 1
          and wl2.items[0]["url"], f"reloaded={len(wl2.items)}")
    removed = wl2.remove(1)
    check("wishlist remove", removed == 1 and len(wl2.items) == 0,
          f"removed={removed}")

    # -- copilot exposes beatport tools when signed in -------------------------
    try:
        from tools.djplanner.copilot import SetCopilot, BEATPORT_TOOLS
        # A "signed-in" client over the fake API.
        bp_client = BP.BeatportClient(auth=cauth, transport=fake_api)
        cp = SetCopilot(lib, theme_name="groove", client=object(),
                        beatport=bp_client, wishlist=BP.Wishlist(tmp))
        names = {t["name"] for t in cp.tools()}
        check("copilot gains beatport tools when signed in",
              {"beatport_search", "beatport_wishlist_add"} <= names,
              f"has {len(names)} tools")
        res = cp.run_tool("beatport_search", {"query": "melodic"})
        check("copilot beatport_search runs", res["count"] == 2
              and "fit_vs_last" not in res["tracks"][0]  # empty set = no cur
              and res["tracks"][0]["library_neighbours"] >= 0,
              f"count={res['count']}")
        # Signed-OUT copilot must NOT expose the tools.
        cp2 = SetCopilot(lib, theme_name="groove", client=object())
        check("copilot hides beatport tools when signed out",
              not ({"beatport_search"} & {t["name"] for t in cp2.tools()}),
              f"tools={len(cp2.tools())}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        check("copilot gains beatport tools when signed in", False,
              f"{type(e).__name__}: {e}")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()

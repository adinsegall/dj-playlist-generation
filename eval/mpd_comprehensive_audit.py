"""
MPD Comprehensive Audit Script

Validates MPD representation models and tests integration with local DJ library.
Combines data integrity checks with practical utility testing.

Author: Adin Segall
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd
from scipy import sparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from gensim.models import Word2Vec
except ImportError:
    print("ERROR: gensim not installed. Run: poetry install")
    sys.exit(1)


class AuditResult:
    """Container for audit results."""

    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.messages: List[str] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def info(self, msg: str) -> None:
        self.messages.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(f"[WARN] {msg}")

    def error(self, msg: str) -> None:
        self.errors.append(f"[ERROR] {msg}")
        self.passed = False

    def success(self, msg: str) -> None:
        self.messages.append(f"[OK] {msg}")

    def print_report(self) -> None:
        status = "PASS" if self.passed else "FAIL"
        print(f"\n{'='*60}")
        print(f"  {self.name}: {status}")
        print(f"{'='*60}")
        for msg in self.messages:
            print(f"  {msg}")
        for warn in self.warnings:
            print(f"  {warn}")
        for err in self.errors:
            print(f"  {err}")


class MPDComprehensiveAudit:
    """Comprehensive audit of MPD representation models."""

    def __init__(
        self,
        models_dir: str = "models",
        data_dir: str = "data"
    ):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)

        # Model references
        self.artist2vec: Optional[Word2Vec] = None
        self.transition_matrix: Optional[sparse.csr_matrix] = None
        self.artist_to_idx: Dict[str, int] = {}
        self.idx_to_artist: Dict[int, str] = {}
        self.vibe_dictionary: Dict = {}
        self.local_tracks: Optional[pd.DataFrame] = None

        # Name↔URI mappings (built by bridge script)
        self.name_to_uri: Dict[str, str] = {}  # lowercase_name -> uri_id
        self.uri_to_name: Dict[str, str] = {}  # uri_id -> original_name
        self.known_uris: Set[str] = set()  # All URIs in MPD

    def load_name_uri_mapping(self) -> bool:
        """Load the artist name→URI mapping from bridge script."""
        # Bridge script outputs: mpd_name_to_uri.json and mpd_known_uris.json
        name_path = self.data_dir / "mpd_name_to_uri.json"
        uris_path = self.data_dir / "mpd_known_uris.json"

        if not name_path.exists():
            return False

        try:
            with open(name_path, 'r') as f:
                self.name_to_uri = json.load(f)

            # Load known URIs if available
            if uris_path.exists():
                with open(uris_path, 'r') as f:
                    self.known_uris = set(json.load(f))

            # Build reverse mapping (uri -> name)
            # Note: multiple names may map to same URI, we keep first
            for name, uri in self.name_to_uri.items():
                if uri not in self.uri_to_name:
                    self.uri_to_name[uri] = name

            return True
        except Exception:
            return False

    def lookup_artist_uri(self, name: str) -> Optional[str]:
        """Look up URI for an artist name."""
        return self.name_to_uri.get(name.lower().strip())

    def lookup_artist_name(self, uri: str) -> str:
        """Look up name for an artist URI, or return truncated URI."""
        # Strip prefix if present
        uri_id = uri[15:] if uri.startswith("spotify:artist:") else uri
        return self.uri_to_name.get(uri_id, uri_id[:20] + "...")

    def run_all_audits(self) -> bool:
        """Run all audit steps in order."""
        print("\n" + "="*60)
        print("  MPD COMPREHENSIVE AUDIT")
        print("="*60)

        # Load name↔URI mapping if available
        if self.load_name_uri_mapping():
            print(f"\n  Loaded name→URI mapping: {len(self.name_to_uri):,} artists")
        else:
            print("\n  [WARN] No name->URI mapping found. Run bridge script first:")
            print("     python -m src.data.bridge_local_to_mpd")

        results = []

        # Step 1: Data Integrity
        result1 = self.audit_data_integrity()
        results.append(result1)
        result1.print_report()

        if not result1.passed:
            print("\n[ERROR] Data integrity failed. Cannot proceed with other audits.")
            return False

        # Step 2: Semantic Accuracy
        result2 = self.audit_semantic_accuracy()
        results.append(result2)
        result2.print_report()

        # Step 3: Local Library Integration
        result3 = self.audit_local_integration()
        results.append(result3)
        result3.print_report()

        # Step 4: Playlist Simulation
        result4 = self.audit_playlist_simulation()
        results.append(result4)
        result4.print_report()

        # Step 5: Visualization
        result5 = self.audit_vibe_visualization()
        results.append(result5)
        result5.print_report()

        # Summary
        print("\n" + "="*60)
        print("  AUDIT SUMMARY")
        print("="*60)
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        print(f"  {passed}/{total} audits passed")

        return all(r.passed for r in results)

    def audit_data_integrity(self) -> AuditResult:
        """Step 1: Verify model files exist and have valid structure."""
        result = AuditResult("1. DATA INTEGRITY CHECK")

        # Check Artist2Vec model
        a2v_path = self.models_dir / "artist2vec.model"
        if not a2v_path.exists():
            result.error(f"Artist2Vec model not found: {a2v_path}")
        else:
            try:
                self.artist2vec = Word2Vec.load(str(a2v_path))
                vocab_size = len(self.artist2vec.wv)
                vector_size = self.artist2vec.wv.vector_size
                result.success(f"Artist2Vec loaded: {vocab_size:,} artists, {vector_size}D vectors")
            except Exception as e:
                result.error(f"Failed to load Artist2Vec: {e}")

        # Check Transition Matrix
        trans_path = self.models_dir / "artist_transitions.npz"
        index_path = self.models_dir / "artist_transitions.index.pkl"

        if not trans_path.exists():
            result.error(f"Transition matrix not found: {trans_path}")
        elif not index_path.exists():
            result.error(f"Transition index not found: {index_path}")
        else:
            try:
                self.transition_matrix = sparse.load_npz(trans_path)
                with open(index_path, 'rb') as f:
                    indices = pickle.load(f)
                    self.artist_to_idx = indices['artist_to_idx']
                    self.idx_to_artist = indices['idx_to_artist']

                n_artists = len(self.artist_to_idx)
                nnz = self.transition_matrix.nnz
                total_possible = n_artists * n_artists
                density = (nnz / total_possible) * 100 if total_possible > 0 else 0

                result.success(f"Transition matrix loaded: {self.transition_matrix.shape}")
                result.info(f"  Artists indexed: {n_artists:,}")
                result.info(f"  Non-zero transitions: {nnz:,}")
                result.info(f"  Density: {density:.6f}% (sparse is good)")

                # Verify dimensions match
                if self.transition_matrix.shape[0] != n_artists:
                    result.error(f"Matrix dimension {self.transition_matrix.shape[0]} != index size {n_artists}")
                else:
                    result.success("Matrix dimensions match index size")

            except Exception as e:
                result.error(f"Failed to load transition matrix: {e}")

        # Check Vibe Dictionary
        vibe_path = self.models_dir / "global_vibe_dictionary.json"
        if not vibe_path.exists():
            result.error(f"Vibe dictionary not found: {vibe_path}")
        else:
            try:
                with open(vibe_path, 'r') as f:
                    self.vibe_dictionary = json.load(f)

                n_keywords = len(self.vibe_dictionary)
                with_audio = sum(1 for v in self.vibe_dictionary.values()
                               if v.get('audio_features') is not None)

                result.success(f"Vibe dictionary loaded: {n_keywords} keywords")
                result.info(f"  Keywords with audio stats: {with_audio}")
                result.info(f"  Keywords without audio stats: {n_keywords - with_audio}")

            except Exception as e:
                result.error(f"Failed to load vibe dictionary: {e}")

        return result

    def audit_semantic_accuracy(self) -> AuditResult:
        """Step 2: Test Artist2Vec and Vibe Dictionary semantics."""
        result = AuditResult("2. SEMANTIC ACCURACY (Vibe Check)")

        if self.artist2vec is None:
            result.error("Artist2Vec not loaded")
            return result

        # Test artist queries
        test_artists = [
            ("Fred again..", ["Four Tet", "Skrillex", "Jamie xx", "Bicep"]),
            ("Chris Lake", ["Fisher", "Green Velvet", "Claude VonStroke"]),
            ("Daft Punk", ["Justice", "Kavinsky", "SebastiAn"]),
        ]

        result.info("Artist2Vec Similarity Tests:")

        # Check if we have the name→URI mapping
        has_mapping = len(self.name_to_uri) > 0
        if has_mapping:
            result.info(f"  (Using name→URI mapping with {len(self.name_to_uri):,} artists)")
        else:
            result.warn("  No name→URI mapping - using fallback URI search")

        for artist_name, expected in test_artists:
            # Try to find the artist URI
            found_uri = None

            # Method 1: Use name→URI mapping (preferred)
            if has_mapping:
                found_uri = self.lookup_artist_uri(artist_name)

            # Method 2: Fallback to searching URIs directly (won't work well)
            if not found_uri:
                for uri in self.artist2vec.wv.key_to_index.keys():
                    if artist_name.lower().replace(" ", "").replace(".", "") in uri.lower():
                        found_uri = uri
                        break

            if found_uri and found_uri in self.artist2vec.wv:
                try:
                    neighbors = self.artist2vec.wv.most_similar(found_uri, topn=5)

                    # Convert neighbor URIs to names if we have mapping
                    if has_mapping:
                        neighbor_names = [self.lookup_artist_name(n[0]) for n in neighbors[:3]]
                        result.success(f"  '{artist_name}' → {neighbor_names}")

                        # Check if any expected artists are in top 10
                        all_neighbors = self.artist2vec.wv.most_similar(found_uri, topn=10)
                        neighbor_names_all = [self.lookup_artist_name(n[0]).lower() for n in all_neighbors]
                        matches = [e for e in expected if e.lower() in neighbor_names_all]
                        if matches:
                            result.info(f"    Expected matches found: {matches}")
                    else:
                        result.info(f"  '{artist_name}' → URIs: {[n[0][:15] for n in neighbors[:3]]}")

                except Exception as e:
                    result.warn(f"  '{artist_name}' query failed: {e}")
            else:
                if found_uri:
                    result.warn(f"  '{artist_name}' mapped to {found_uri[:20]}... but not in model")
                else:
                    result.warn(f"  '{artist_name}' not found in mapping")

        # Test Vibe Dictionary
        result.info("\nVibe Dictionary Tests:")

        vibe_tests = ["berghain", "opening", "peak time", "deep house", "chill"]

        for keyword in vibe_tests:
            if keyword in self.vibe_dictionary:
                vibe = self.vibe_dictionary[keyword]
                track_count = vibe.get('total_tracks_in_mpd', 0)
                audio = vibe.get('audio_features')

                if audio:
                    bpm = audio.get('bpm', {}).get('mean', 'N/A')
                    energy = audio.get('energy', {}).get('mean', 'N/A')
                    if isinstance(bpm, float):
                        bpm = f"{bpm:.1f}"
                    if isinstance(energy, float):
                        energy = f"{energy:.2f}"
                    result.success(f"  '{keyword}': {track_count:,} tracks, BPM={bpm}, Energy={energy}")
                else:
                    result.info(f"  '{keyword}': {track_count:,} tracks (no audio stats)")
            else:
                result.warn(f"  '{keyword}' not in vibe dictionary")

        # Compare opening vs peak time energy
        if "opening" in self.vibe_dictionary and "peak time" in self.vibe_dictionary:
            open_audio = self.vibe_dictionary["opening"].get("audio_features")
            peak_audio = self.vibe_dictionary["peak time"].get("audio_features")

            if open_audio and peak_audio:
                open_energy = open_audio.get("energy", {}).get("mean", 0)
                peak_energy = peak_audio.get("energy", {}).get("mean", 0)

                if peak_energy > open_energy:
                    result.success(f"  Energy progression correct: opening ({open_energy:.2f}) < peak ({peak_energy:.2f})")
                else:
                    result.warn(f"  Unexpected: opening ({open_energy:.2f}) >= peak ({peak_energy:.2f})")

        return result

    def audit_local_integration(self) -> AuditResult:
        """Step 3: Test integration with local library."""
        result = AuditResult("3. LOCAL LIBRARY INTEGRATION")

        # Load local tracks
        tracks_path = self.data_dir / "tracks_master.parquet"
        if not tracks_path.exists():
            result.error(f"Local tracks not found: {tracks_path}")
            return result

        try:
            self.local_tracks = pd.read_parquet(tracks_path)
            result.success(f"Loaded {len(self.local_tracks):,} local tracks")
        except Exception as e:
            result.error(f"Failed to load local tracks: {e}")
            return result

        # Check for Spotify URI columns
        spotify_uri_col = None
        for col in ['spotify_uri', 'spotify_id', 'uri']:
            if col in self.local_tracks.columns:
                spotify_uri_col = col
                break

        # Check for artist URI/name
        artist_col = None
        for col in ['artist', 'artist_name', 'spotify_artist']:
            if col in self.local_tracks.columns:
                artist_col = col
                break

        if artist_col is None:
            result.warn("No artist column found in local tracks")
            return result

        result.info(f"  Using artist column: '{artist_col}'")

        # Get unique local artists
        local_artists = set(self.local_tracks[artist_col].dropna().unique())
        result.info(f"  Unique local artists: {len(local_artists):,}")

        # Coverage Score: Use mpd_uri column created by bridge script
        # Bridge already did the matching (direct, name, proxy)
        if 'mpd_uri' in self.local_tracks.columns:
            total_tracks = len(self.local_tracks)
            matched_tracks = self.local_tracks['mpd_uri'].notna().sum()
            coverage = (matched_tracks / total_tracks) * 100

            result.success(f"  Bridge Coverage: {matched_tracks}/{total_tracks} tracks ({coverage:.1f}%)")

            # Show breakdown by method if available
            if 'mpd_method' in self.local_tracks.columns:
                method_counts = self.local_tracks['mpd_method'].value_counts()
                result.info("  Matching breakdown:")
                for method, count in method_counts.items():
                    pct = (count / total_tracks) * 100
                    label = {'direct': '[DIRECT]', 'name': '[NAME]', 'proxy': '[PROXY]', 'failed': '[FAILED]'}.get(method, '-')
                    result.info(f"    {label} {method}: {count} ({pct:.1f}%)")

            # Check how many matched URIs are in Artist2Vec
            if self.artist2vec:
                matched_uris = set(self.local_tracks['mpd_uri'].dropna().unique())
                model_uris = set(self.artist2vec.wv.key_to_index.keys())
                in_model = matched_uris & model_uris
                model_coverage = (len(in_model) / len(matched_uris)) * 100 if matched_uris else 0
                result.info(f"  URIs in Artist2Vec model: {len(in_model)}/{len(matched_uris)} ({model_coverage:.1f}%)")
        else:
            result.warn("No mpd_uri column found. Run bridge script first:")
            result.info("  python -m src.data.bridge_local_to_mpd")

        # For compatibility, set spotify_artist_col to mpd_uri
        spotify_artist_col = 'mpd_uri' if 'mpd_uri' in self.local_tracks.columns else None

        # Transition Potential Test
        if self.transition_matrix is not None and len(self.artist_to_idx) > 0:
            result.info("\nTransition Potential Test:")

            # Find a local artist that exists in transition matrix
            # mpd_uri values are already normalized (no prefix)
            test_artist = None
            test_artist_name = None

            if spotify_artist_col:
                # Build set of local URIs for fast lookup
                local_uri_set = set(self.local_tracks[spotify_artist_col].dropna().unique())

                for _, track_row in self.local_tracks.head(100).iterrows():
                    uri = track_row.get(spotify_artist_col)
                    if uri and uri in self.artist_to_idx:
                        test_artist = uri
                        test_artist_name = track_row.get(artist_col, "Unknown")
                        break

            if test_artist:
                result.info(f"  Testing transitions from: {test_artist_name}")

                idx = self.artist_to_idx[test_artist]
                trans_row = self.transition_matrix.getrow(idx).toarray().flatten()
                top_indices = np.argsort(trans_row)[-10:][::-1]

                result.info(f"  Top transition targets (in local library):")
                found_in_library = 0

                for i in top_indices:
                    if trans_row[i] > 0:
                        target_uri = self.idx_to_artist[i]
                        prob = trans_row[i]

                        # Check if target is in local library
                        if target_uri in local_uri_set:
                            found_in_library += 1
                            # Get artist name from uri_to_name mapping or local tracks
                            target_name = self.uri_to_name.get(target_uri, target_uri[:20])
                            if found_in_library <= 5:
                                result.info(f"    → {target_name}: {prob:.3f}")

                if found_in_library == 0:
                    result.warn("  No transition targets found in local library")
                else:
                    result.success(f"  Found {found_in_library} transition targets in library")
            else:
                result.warn("  No local artist found in transition matrix")

        return result

    def audit_playlist_simulation(self) -> AuditResult:
        """Step 4: Simulate a 3-track playlist path."""
        result = AuditResult("4. PLAYLIST SIMULATION (Product Demo)")

        if self.local_tracks is None:
            result.error("Local tracks not loaded")
            return result

        if self.artist2vec is None or self.transition_matrix is None:
            result.error("Models not loaded")
            return result

        # Find artist column
        artist_col = None
        for col in ['artist', 'artist_name', 'spotify_artist']:
            if col in self.local_tracks.columns:
                artist_col = col
                break

        if not artist_col:
            result.warn("No artist name column found")
            return result

        # Use mpd_uri column from bridge script
        if 'mpd_uri' not in self.local_tracks.columns:
            result.warn("No mpd_uri column found. Run bridge script first:")
            result.info("  python -m src.data.bridge_local_to_mpd")
            return result

        result.info("Simulating 3-step playlist path:")

        # Build set of local URIs for fast lookup
        local_uri_set = set(self.local_tracks['mpd_uri'].dropna().unique())

        # Find starting artist in local library that exists in models
        start_artist_uri = None
        start_artist_name = None

        for _, row in self.local_tracks.iterrows():
            uri = row.get('mpd_uri')
            if uri and uri in self.artist2vec.wv and uri in self.artist_to_idx:
                start_artist_uri = uri
                start_artist_name = row.get(artist_col, "Unknown")
                break

        if not start_artist_uri:
            result.warn("No local artist found in both models")
            return result

        result.info(f"\n  START: {start_artist_name}")

        # Step 1: Use Artist2Vec similarity
        try:
            similar = self.artist2vec.wv.most_similar(start_artist_uri, topn=20)

            # Find one that's in local library
            step1_uri = None
            step1_name = None

            for uri, score in similar:
                if uri in local_uri_set:
                    step1_uri = uri
                    step1_name = self.uri_to_name.get(uri, uri[:20])
                    result.info(f"  → STEP 1 (similarity): {step1_name} (score: {score:.3f})")
                    break

            if not step1_uri:
                result.warn("  No similar artist found in local library")
                return result

        except Exception as e:
            result.error(f"  Artist2Vec query failed: {e}")
            return result

        # Step 2: Use Transition Matrix
        step2_name = None
        if step1_uri in self.artist_to_idx:
            idx = self.artist_to_idx[step1_uri]
            trans_row = self.transition_matrix.getrow(idx).toarray().flatten()
            top_indices = np.argsort(trans_row)[-20:][::-1]

            for i in top_indices:
                if trans_row[i] > 0:
                    target_uri = self.idx_to_artist[i]

                    # Check if in local library (and not same as step1)
                    if target_uri != step1_uri and target_uri in local_uri_set:
                        step2_name = self.uri_to_name.get(target_uri, target_uri[:20])
                        result.info(f"  → STEP 2 (transition): {step2_name} (prob: {trans_row[i]:.3f})")
                        break

            if not step2_name:
                result.warn("  No transition target found in local library")
        else:
            result.warn(f"  {step1_name} not in transition matrix")

        result.success("\n  Path: {} → {} → {}".format(
            start_artist_name,
            step1_name or "???",
            step2_name or "???"
        ))

        return result

    def audit_vibe_visualization(self) -> AuditResult:
        """Step 5: ASCII table of vibe comparisons."""
        result = AuditResult("5. VIBE COMPARISON (Visualization)")

        if not self.vibe_dictionary:
            result.error("Vibe dictionary not loaded")
            return result

        # Keywords to compare
        compare_keywords = ["chill", "party", "workout", "deep house", "techno"]

        # Build table data
        table_data = []
        for keyword in compare_keywords:
            if keyword in self.vibe_dictionary:
                vibe = self.vibe_dictionary[keyword]
                audio = vibe.get("audio_features") or {}

                bpm = audio.get("bpm", {}).get("mean")
                energy = audio.get("energy", {}).get("mean")
                valence = audio.get("valence", {}).get("mean")
                dance = audio.get("danceability", {}).get("mean")
                tracks = vibe.get("total_tracks_in_mpd", 0)

                table_data.append({
                    "keyword": keyword,
                    "tracks": tracks,
                    "bpm": f"{bpm:.0f}" if bpm else "-",
                    "energy": f"{energy:.2f}" if energy else "-",
                    "valence": f"{valence:.2f}" if valence else "-",
                    "dance": f"{dance:.2f}" if dance else "-",
                })

        if not table_data:
            result.warn("No keywords found for comparison")
            return result

        # Print ASCII table
        result.info("\n  Vibe Audio Feature Comparison:")
        result.info("  " + "-" * 62)
        result.info("  | {:12} | {:>8} | {:>6} | {:>6} | {:>6} | {:>6} |".format(
            "Keyword", "Tracks", "BPM", "Energy", "Valence", "Dance"
        ))
        result.info("  " + "-" * 62)

        for row in table_data:
            result.info("  | {:12} | {:>8} | {:>6} | {:>6} | {:>6} | {:>6} |".format(
                row["keyword"][:12],
                f"{row['tracks']:,}" if isinstance(row['tracks'], int) else row['tracks'],
                row["bpm"],
                row["energy"],
                row["valence"],
                row["dance"]
            ))

        result.info("  " + "-" * 62)

        # Insights
        result.info("\n  Insights:")

        # Find highest/lowest energy
        with_energy = [r for r in table_data if r["energy"] != "-"]
        if with_energy:
            highest = max(with_energy, key=lambda x: float(x["energy"]))
            lowest = min(with_energy, key=lambda x: float(x["energy"]))
            result.info(f"  • Highest energy: {highest['keyword']} ({highest['energy']})")
            result.info(f"  • Lowest energy: {lowest['keyword']} ({lowest['energy']})")

        # Find fastest BPM
        with_bpm = [r for r in table_data if r["bpm"] != "-"]
        if with_bpm:
            fastest = max(with_bpm, key=lambda x: float(x["bpm"]))
            result.info(f"  • Fastest tempo: {fastest['keyword']} ({fastest['bpm']} BPM)")

        return result


def main():
    """Run the comprehensive audit."""
    import argparse

    parser = argparse.ArgumentParser(description="MPD Comprehensive Audit")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    parser.add_argument("--data-dir", default="data", help="Data directory")

    args = parser.parse_args()

    audit = MPDComprehensiveAudit(
        models_dir=args.models_dir,
        data_dir=args.data_dir
    )

    success = audit.run_all_audits()

    print("\n" + "="*60)
    if success:
        print("  [OK] ALL AUDITS PASSED")
    else:
        print("  [FAIL] SOME AUDITS FAILED")
    print("="*60 + "\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

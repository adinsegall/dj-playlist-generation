"""
User Study Data Analysis for DJ Playlist Generator Evaluation.

Aggregates and analyzes participant data from the user study:
- Survey responses (Likert scales, qualitative feedback)
- Per-playlist feedback (satisfaction ratings, comments)
- Interaction logs (queries, timestamps, actions)
- Playlist quality metrics

Author: Adin Segall
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
USER_STUDY_DIR = PROJECT_ROOT / "output" / "user_studies" / "participants"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# Likert question labels (matching app.py)
LIKERT_QUESTIONS = [
    "The system accurately interpreted my prompts",
    "Generated playlists had good track selection for the specified context",
    "The track ordering/progression was musically coherent",
    "Energy transitions were smooth and appropriate",
    "The system understood genre specifications correctly",
    "I could achieve my desired results through prompt refinement",
    "This tool would save me time in playlist preparation",
    "I would use this for professional gig preparation",
    "I trust the AI's music recommendations",
    "The playlists were appropriately diverse"
]

# System handling categories
SYSTEM_HANDLING_CATEGORIES = [
    "multi_genre_requests",
    "venue_understanding",
    "mixing_bpm_key",
    "energy_management"
]


@dataclass
class ParticipantData:
    """Container for a single participant's data."""
    participant_id: str
    name: str
    session_id: str
    profile: Dict[str, Any] = field(default_factory=dict)
    survey: Optional[Dict[str, Any]] = None
    playlist_feedback: List[Dict[str, Any]] = field(default_factory=list)
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    playlists_generated: int = 0
    queries: List[str] = field(default_factory=list)


@dataclass
class AggregatedResults:
    """Container for aggregated study results.

    The aggregator supports BOTH the v1 schema (May 2 self-test) and the
    v2 paired-Likert schema (2026-05-05+). v2 fields are populated from
    the per-query `playlist_NNN_feedback.json` files (see app.py
    save_paired_feedback) and the much-simpler post-study survey.
    """
    n_participants: int = 0
    n_surveys_completed: int = 0
    n_playlists_total: int = 0
    n_feedback_entries: int = 0

    # ── v1 schema (kept for backward compat with May 2 self-test) ─────
    likert_scores: Dict[str, List[int]] = field(default_factory=dict)
    system_handling_scores: Dict[str, List[int]] = field(default_factory=dict)
    pct_appropriate: List[int] = field(default_factory=list)
    gig_ready_counts: List[int] = field(default_factory=list)
    pct_reordering: List[int] = field(default_factory=list)
    generation_time_comparison: List[str] = field(default_factory=list)
    satisfaction_ratings: List[int] = field(default_factory=list)

    # ── v2/v3 schema — per-query paired Likerts + track marks ────────
    # Each list entry is a single (participant, query) datapoint. The
    # method dimension is captured in the keys 'heuristic'/'embedding'.
    # v3 (2026-05-14) renamed 'cohesion' to 'playlist_quality' (Q4 reworded
    # from "does it flow as a DJ set" to "does this work as a Rekordbox
    # playlist"). For back-compat both keys feed the same accumulator —
    # downstream analysis treats them as the same Likert dimension.
    relevance_per_method: Dict[str, List[int]] = field(default_factory=dict)
    playlist_quality_per_method: Dict[str, List[int]] = field(default_factory=dict)
    context_appropriate_per_method: Dict[str, List[str]] = field(default_factory=dict)
    n_dont_belong_per_method: Dict[str, List[int]] = field(default_factory=dict)
    n_keep_per_method: Dict[str, List[int]] = field(default_factory=dict)
    n_tracks_total_per_method: Dict[str, List[int]] = field(default_factory=dict)
    forced_preference_per_query: List[Dict[str, Any]] = field(default_factory=list)
    preference_reasons: List[str] = field(default_factory=list)
    cross_query_summaries: List[str] = field(default_factory=list)
    query_type_performance: List[str] = field(default_factory=list)
    overall_picks: Counter = field(default_factory=Counter)

    # Qualitative data — shared between v1 and v2
    what_worked: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    feature_requests: List[str] = field(default_factory=list)
    genre_issues: List[str] = field(default_factory=list)
    other_comments: List[str] = field(default_factory=list)

    # Usage stats
    used_for_gigs: Counter = field(default_factory=Counter)
    would_recommend: Counter = field(default_factory=Counter)
    experience_levels: Counter = field(default_factory=Counter)

    # All queries for analysis
    all_queries: List[str] = field(default_factory=list)


class UserStudyAnalyzer:
    """Analyze user study data from all participants."""

    def __init__(self, study_dir: Optional[Path] = None):
        """
        Initialize analyzer.

        Args:
            study_dir: Path to user study participants directory
        """
        self.study_dir = study_dir or USER_STUDY_DIR
        self.participants: List[ParticipantData] = []
        self.results: Optional[AggregatedResults] = None

    def load_all_participants(self) -> List[ParticipantData]:
        """Load data for all participants."""
        self.participants = []

        if not self.study_dir.exists():
            logger.warning(f"Study directory not found: {self.study_dir}")
            return []

        # Find all participant directories
        participant_dirs = [d for d in self.study_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(participant_dirs)} participant directories")

        for pdir in participant_dirs:
            try:
                participant = self._load_participant(pdir)
                if participant:
                    self.participants.append(participant)
                    logger.info(f"Loaded: {participant.name} ({participant.playlists_generated} playlists)")
            except Exception as e:
                logger.error(f"Failed to load {pdir.name}: {e}")

        logger.info(f"Successfully loaded {len(self.participants)} participants")
        return self.participants

    def _load_participant(self, pdir: Path) -> Optional[ParticipantData]:
        """Load a single participant's data."""
        # Extract session ID from directory name (format: name_sessionid)
        parts = pdir.name.rsplit('_', 1)
        if len(parts) != 2:
            logger.warning(f"Invalid directory name format: {pdir.name}")
            return None

        name_sanitized, session_id = parts

        participant = ParticipantData(
            participant_id=pdir.name,
            name=name_sanitized.replace('_', ' ').title(),
            session_id=session_id
        )

        # Load profile
        profile_path = pdir / "participant_profile.json"
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                participant.profile = json.load(f)
                participant.name = participant.profile.get('name', participant.name)

        # Load session state for queries
        session_path = pdir / "session_state.json"
        if session_path.exists():
            with open(session_path, 'r') as f:
                session = json.load(f)
                participant.queries = session.get('all_queries', [])
                participant.playlists_generated = session.get('playlist_count', 0)

        # Load survey
        survey_path = pdir / "final_survey.json"
        if survey_path.exists():
            with open(survey_path, 'r') as f:
                participant.survey = json.load(f)

        # Load interactions
        interactions_path = pdir / "interactions.json"
        if interactions_path.exists():
            with open(interactions_path, 'r') as f:
                participant.interactions = json.load(f)

        # Load playlist feedback
        playlists_dir = pdir / "playlists"
        if playlists_dir.exists():
            for feedback_file in playlists_dir.glob("*_feedback.json"):
                try:
                    with open(feedback_file, 'r') as f:
                        feedback = json.load(f)
                        participant.playlist_feedback.append(feedback)
                except Exception as e:
                    logger.warning(f"Failed to load feedback {feedback_file}: {e}")

        return participant

    def aggregate_results(self) -> AggregatedResults:
        """Aggregate results across all participants."""
        if not self.participants:
            self.load_all_participants()

        results = AggregatedResults()
        results.n_participants = len(self.participants)

        for p in self.participants:
            # Count playlists
            results.n_playlists_total += p.playlists_generated
            results.all_queries.extend(p.queries)

            # Experience level
            if p.profile:
                exp = p.profile.get('years_experience', 'Unknown')
                results.experience_levels[exp] += 1

            # Process survey
            if p.survey:
                results.n_surveys_completed += 1
                self._process_survey(p.survey, results)

            # Process playlist feedback (v1 + v2)
            for fb in p.playlist_feedback:
                results.n_feedback_entries += 1
                self._process_playlist_feedback(fb, results)

        self.results = results
        return results

    def _process_playlist_feedback(self, fb: Dict, results: AggregatedResults):
        """Process one playlist's paired feedback (v1 + v2 schemas).

        v2 schema (2026-05-05+) writes per-method paired Likerts on
        relevance (Q1) and cohesion (Q4), per-method context_appropriate
        (Q5), per-method don't-belong / keep track ID lists (Q2/Q3),
        a forced-preference choice (Q6), and a free-text reason (Q7).

        v1 schema (May 2 self-test) wrote a single 'satisfaction' int
        and a 'ratings' dict mapping method → satisfaction Likert.
        """
        # Detect v2 by presence of `per_method` key.
        per_method = fb.get('per_method')
        if isinstance(per_method, dict) and per_method:
            # v2 path. The keys are internal method names ('heuristic', 'embedding').
            for m, v in per_method.items():
                if not isinstance(v, dict):
                    continue
                if 'relevance' in v:
                    results.relevance_per_method.setdefault(m, []).append(int(v['relevance']))
                # v3 schema key is 'playlist_quality'; v2 key was 'cohesion'.
                # Both map into the same accumulator for analysis continuity.
                if 'playlist_quality' in v:
                    results.playlist_quality_per_method.setdefault(m, []).append(
                        int(v['playlist_quality'])
                    )
                elif 'cohesion' in v:
                    results.playlist_quality_per_method.setdefault(m, []).append(
                        int(v['cohesion'])
                    )
                if 'context_appropriate' in v:
                    results.context_appropriate_per_method.setdefault(m, []).append(
                        str(v['context_appropriate'])
                    )
                if 'n_dont_belong' in v:
                    results.n_dont_belong_per_method.setdefault(m, []).append(
                        int(v['n_dont_belong'])
                    )
                if 'n_keep' in v:
                    results.n_keep_per_method.setdefault(m, []).append(int(v['n_keep']))
                if 'n_tracks_total' in v:
                    results.n_tracks_total_per_method.setdefault(m, []).append(
                        int(v['n_tracks_total'])
                    )
            forced = fb.get('forced_preference')
            if forced:
                results.forced_preference_per_query.append({
                    'query': fb.get('query', ''),
                    'forced_preference': forced,
                    'method_label_map': fb.get('method_label_map'),
                    'shown_labels': fb.get('shown_labels'),
                })
            if fb.get('preference_reason'):
                results.preference_reasons.append(fb['preference_reason'])
            return

        # v1 path: single 'satisfaction' int (legacy) OR 'ratings' dict
        # mapping method → relevance Likert. Both routed into the legacy
        # `satisfaction_ratings` list so existing analysis works.
        if 'satisfaction' in fb:
            results.satisfaction_ratings.append(fb['satisfaction'])
        elif 'ratings' in fb and isinstance(fb['ratings'], dict):
            for v in fb['ratings'].values():
                try:
                    results.satisfaction_ratings.append(int(v))
                except (TypeError, ValueError):
                    continue

    def _process_survey(self, survey: Dict, results: AggregatedResults):
        """Process a single survey into aggregated results.

        Branches on `schema_version`. v2 surveys (2026-05-05+) carry
        cross-query free-text Q8/Q9 + a forced overall pick + optional
        feature_requests/other_comments. v1 surveys (May 2 self-test)
        carry the older `likert_responses`/`quality_assessment`/etc.
        """
        schema_version = survey.get('schema_version', '')
        is_v2 = schema_version.startswith('v2_paired_likert')

        if is_v2:
            # ── v2 schema ──────────────────────────────────────────────
            if survey.get('cross_query_summary'):
                results.cross_query_summaries.append(survey['cross_query_summary'])
            if survey.get('query_type_performance'):
                results.query_type_performance.append(survey['query_type_performance'])
            if survey.get('overall_pick'):
                results.overall_picks[survey['overall_pick']] += 1
            if survey.get('feature_requests'):
                results.feature_requests.append(survey['feature_requests'])
            if survey.get('other_comments'):
                results.other_comments.append(survey['other_comments'])
            return

        # ── v1 schema ──────────────────────────────────────────────────
        # Likert responses
        likert = survey.get('likert_responses', {})
        for i, question in enumerate(LIKERT_QUESTIONS):
            key = f"q{i+1}"
            if key in likert:
                if question not in results.likert_scores:
                    results.likert_scores[question] = []
                results.likert_scores[question].append(likert[key])

        # System handling
        handling = survey.get('system_handling', {})
        for cat in SYSTEM_HANDLING_CATEGORIES:
            if cat in handling:
                if cat not in results.system_handling_scores:
                    results.system_handling_scores[cat] = []
                results.system_handling_scores[cat].append(handling[cat])

        # Quality assessment
        quality = survey.get('quality_assessment', {})
        if 'pct_appropriate' in quality:
            results.pct_appropriate.append(quality['pct_appropriate'])
        if 'gig_ready_count' in quality:
            results.gig_ready_counts.append(quality['gig_ready_count'])
        if 'pct_reordering' in quality:
            results.pct_reordering.append(quality['pct_reordering'])
        if 'generation_time_comparison' in quality:
            results.generation_time_comparison.append(quality['generation_time_comparison'])

        # Usage
        usage = survey.get('usage', {})
        if 'used_for_gigs' in usage:
            results.used_for_gigs[usage['used_for_gigs']] += 1

        # Qualitative
        qual = survey.get('qualitative', {})
        if qual.get('what_worked'):
            results.what_worked.append(qual['what_worked'])
        if qual.get('limitations'):
            results.limitations.append(qual['limitations'])
        if qual.get('feature_requests'):
            results.feature_requests.append(qual['feature_requests'])
        if qual.get('genre_issues'):
            results.genre_issues.append(qual['genre_issues'])
        if qual.get('other_comments'):
            results.other_comments.append(qual['other_comments'])
        if 'would_recommend' in qual:
            results.would_recommend[qual['would_recommend']] += 1

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics from aggregated results."""
        if not self.results:
            self.aggregate_results()

        r = self.results
        stats = {
            'overview': {
                'n_participants': r.n_participants,
                'n_surveys_completed': r.n_surveys_completed,
                'n_playlists_total': r.n_playlists_total,
                'n_feedback_entries': r.n_feedback_entries,
                'avg_playlists_per_participant': r.n_playlists_total / max(r.n_participants, 1)
            },
            'likert_scores': {},
            'system_handling': {},
            'quality_assessment': {},
            'satisfaction': {},
            'usage': {},
            'experience_distribution': dict(r.experience_levels)
        }

        # Likert statistics
        for question, scores in r.likert_scores.items():
            if scores:
                stats['likert_scores'][question] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'median': float(np.median(scores)),
                    'min': int(min(scores)),
                    'max': int(max(scores)),
                    'n': len(scores),
                    'distribution': dict(Counter(scores))
                }

        # System handling statistics
        for cat, scores in r.system_handling_scores.items():
            if scores:
                stats['system_handling'][cat] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'median': float(np.median(scores)),
                    'n': len(scores)
                }

        # Quality assessment
        if r.pct_appropriate:
            stats['quality_assessment']['pct_appropriate'] = {
                'mean': float(np.mean(r.pct_appropriate)),
                'std': float(np.std(r.pct_appropriate)),
                'median': float(np.median(r.pct_appropriate))
            }
        if r.pct_reordering:
            stats['quality_assessment']['pct_reordering'] = {
                'mean': float(np.mean(r.pct_reordering)),
                'std': float(np.std(r.pct_reordering)),
                'median': float(np.median(r.pct_reordering))
            }
        stats['quality_assessment']['generation_time_comparison'] = dict(Counter(r.generation_time_comparison))

        # Satisfaction
        if r.satisfaction_ratings:
            stats['satisfaction'] = {
                'mean': float(np.mean(r.satisfaction_ratings)),
                'std': float(np.std(r.satisfaction_ratings)),
                'median': float(np.median(r.satisfaction_ratings)),
                'distribution': dict(Counter(r.satisfaction_ratings))
            }

        # Usage
        stats['usage']['used_for_gigs'] = dict(r.used_for_gigs)
        stats['usage']['would_recommend'] = dict(r.would_recommend)

        return stats

    def export_results(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Export all results to files."""
        output_dir = output_dir or EVALUATION_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Compute statistics
        stats = self.compute_statistics()

        # Export JSON
        json_path = output_dir / "user_study_results.json"
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        paths['json'] = json_path
        logger.info(f"Exported statistics to {json_path}")

        # Export qualitative data
        qual_path = output_dir / "user_study_qualitative.json"
        qual_data = {
            'what_worked': self.results.what_worked,
            'limitations': self.results.limitations,
            'feature_requests': self.results.feature_requests,
            'genre_issues': self.results.genre_issues,
            'other_comments': self.results.other_comments,
            'all_queries': self.results.all_queries
        }
        with open(qual_path, 'w') as f:
            json.dump(qual_data, f, indent=2)
        paths['qualitative'] = qual_path
        logger.info(f"Exported qualitative data to {qual_path}")

        # Export Likert scores as CSV for easy analysis
        if self.results.likert_scores:
            likert_data = []
            for q, scores in self.results.likert_scores.items():
                for score in scores:
                    likert_data.append({'question': q, 'score': score})
            likert_df = pd.DataFrame(likert_data)
            csv_path = output_dir / "user_study_likert.csv"
            likert_df.to_csv(csv_path, index=False)
            paths['likert_csv'] = csv_path
            logger.info(f"Exported Likert data to {csv_path}")

        return paths

    def print_summary(self):
        """Print a human-readable summary of results."""
        stats = self.compute_statistics()

        print("\n" + "="*60)
        print("USER STUDY RESULTS SUMMARY")
        print("="*60)

        # Overview
        overview = stats['overview']
        print(f"\nParticipants: {overview['n_participants']}")
        print(f"Surveys completed: {overview['n_surveys_completed']}")
        print(f"Total playlists generated: {overview['n_playlists_total']}")
        print(f"Avg playlists per participant: {overview['avg_playlists_per_participant']:.1f}")

        # Likert scores
        print("\n" + "-"*60)
        print("LIKERT SCORES (1-5 scale)")
        print("-"*60)
        for question, data in stats['likert_scores'].items():
            short_q = question[:50] + "..." if len(question) > 50 else question
            print(f"{short_q}")
            print(f"  Mean: {data['mean']:.2f} (std: {data['std']:.2f})")

        # System handling
        if stats['system_handling']:
            print("\n" + "-"*60)
            print("SYSTEM HANDLING SCORES (1-5 scale)")
            print("-"*60)
            for cat, data in stats['system_handling'].items():
                print(f"{cat}: {data['mean']:.2f} (std: {data['std']:.2f})")

        # Satisfaction
        if stats['satisfaction']:
            print("\n" + "-"*60)
            print("OVERALL SATISFACTION")
            print("-"*60)
            sat = stats['satisfaction']
            print(f"Mean: {sat['mean']:.2f} (std: {sat['std']:.2f})")
            print(f"Distribution: {sat['distribution']}")

        # Quality
        print("\n" + "-"*60)
        print("QUALITY ASSESSMENT")
        print("-"*60)
        qual = stats['quality_assessment']
        if 'pct_appropriate' in qual:
            print(f"% tracks appropriate: {qual['pct_appropriate']['mean']:.1f}%")
        if 'pct_reordering' in qual:
            print(f"% requiring reordering: {qual['pct_reordering']['mean']:.1f}%")
        print(f"Generation time comparison: {qual.get('generation_time_comparison', {})}")

        # Usage
        print("\n" + "-"*60)
        print("USAGE")
        print("-"*60)
        print(f"Used for gigs: {stats['usage']['used_for_gigs']}")
        print(f"Would recommend: {stats['usage']['would_recommend']}")

        print("\n" + "="*60)


def main():
    """Run user study analysis."""
    print("DJ Playlist Generator - User Study Analysis")
    print("="*50)

    analyzer = UserStudyAnalyzer()

    # Load all participants
    participants = analyzer.load_all_participants()

    if not participants:
        print("\nNo participant data found yet.")
        print(f"Looking in: {USER_STUDY_DIR}")
        print("\nWaiting for user study data to be collected...")
        return

    # Aggregate and analyze
    analyzer.aggregate_results()

    # Print summary
    analyzer.print_summary()

    # Export results
    paths = analyzer.export_results()
    print("\nExported files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

"""
DJ Playlist Request Schemas

Pydantic v2 models defining the data contract between NLP parsing
and playlist generation. Captures strict math (BPM, duration),
energy bounds, and fuzzy intent (vibes, artists).

Author: Adin Segall
"""

from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class EnergyProfile(str, Enum):
    """Energy curve shape over the playlist duration."""
    RAMP_UP = "ramp_up"          # Start low, end high (opening set)
    RAMP_DOWN = "ramp_down"      # Start high, end low (closing set)
    MOUNTAIN = "mountain"        # Low → peak → low (full journey)
    VALLEY = "valley"            # High → dip → high (energy break)
    STEADY_HIGH = "steady_high"  # Maintain high energy throughout
    STEADY_LOW = "steady_low"    # Maintain low energy throughout
    WAVE = "wave"                # Oscillating energy (multiple peaks)


class VenueStyle(str, Enum):
    """Venue archetype affecting track selection and energy."""
    # Underground/Industrial
    BERGHAIN = "berghain"        # Dark, hypnotic, industrial techno
    TRESOR = "tresor"            # Raw, uncompromising, hard techno
    WAREHOUSE = "warehouse"      # Industrial, gritty, no frills

    # Quality Underground
    FABRIC = "fabric"            # UK bass, diverse, quality-focused
    SMARTBAR = "smartbar"        # Chicago house/techno heritage
    CLUB_SPACE = "club_space"    # Marathon sets, sunrise energy

    # Ibiza Style
    DC10 = "dc10"                # Circoloco vibes, groovy techno
    AMNESIA = "amnesia"          # Peak time, euphoric, big room
    HI_IBIZA = "hi_ibiza"        # Modern, production-heavy

    # Festival
    TOMORROWLAND = "tomorrowland"  # Mainstage energy, crowd-pleasing
    MOVEMENT = "movement"          # Detroit heritage, authentic
    BURNING_MAN = "burning_man"    # Experimental, sunrise sets

    # Radio/Stream
    ESSENTIAL_MIX = "essential_mix"  # Curated journey, showcase
    BOILER_ROOM = "boiler_room"      # Intimate, credibility-focused

    # Generic
    CLUB = "club"                # Standard nightclub
    FESTIVAL = "festival"        # Standard festival
    BEACH = "beach"              # Daytime, relaxed, sunny


# Venue energy modifiers - affects how energy levels are interpreted
VENUE_ENERGY_MODIFIERS = {
    VenueStyle.BERGHAIN: 0.85,      # Darker, more restrained peaks
    VenueStyle.TRESOR: 0.80,        # Raw, unpolished energy
    VenueStyle.WAREHOUSE: 0.85,     # Industrial, darker
    VenueStyle.FABRIC: 0.90,        # Slightly darker
    VenueStyle.SMARTBAR: 0.90,      # Understated cool
    VenueStyle.CLUB_SPACE: 1.0,     # Full energy range
    VenueStyle.DC10: 0.95,          # Groovy, not peak
    VenueStyle.AMNESIA: 1.1,        # Big room energy
    VenueStyle.HI_IBIZA: 1.05,      # Modern production
    VenueStyle.TOMORROWLAND: 1.15,  # Mainstage energy
    VenueStyle.MOVEMENT: 0.90,      # Authentic Detroit
    VenueStyle.BURNING_MAN: 0.95,   # Experimental
    VenueStyle.ESSENTIAL_MIX: 1.0,  # Full range showcase
    VenueStyle.BOILER_ROOM: 0.90,   # Credibility > hype
    VenueStyle.CLUB: 1.0,           # Standard
    VenueStyle.FESTIVAL: 1.1,       # Festival energy
    VenueStyle.BEACH: 0.90,         # Daytime relaxed
}


class Specifics(BaseModel):
    """Explicit artist/track requests from the user."""

    included_artists: List[str] = Field(
        default_factory=list,
        description="Artists the user explicitly wants to include (e.g., 'play some Drake')"
    )
    excluded_artists: List[str] = Field(
        default_factory=list,
        description="Artists the user explicitly wants to exclude (e.g., 'no mainstream')"
    )
    reference_tracks: List[str] = Field(
        default_factory=list,
        description="Specific tracks mentioned as reference points"
    )
    label_artist_query: Optional[str] = Field(
        default=None,
        description="Label name when user wants 'all tracks from artists on X label' (e.g., 'solid grooves', 'drumcode'). This is RESTRICTIVE - only show tracks by artists on this label."
    )


class VibeProfile(BaseModel):
    """Fuzzy intent captured as keywords and genre weights."""

    keywords: List[str] = Field(
        default_factory=list,
        description="Vibe keywords for vector search (e.g., 'dark', 'hypnotic', 'groovy')"
    )
    genre_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Genre distribution (e.g., {'Techno': 0.8, 'House': 0.2})"
    )

    @field_validator('genre_weights')
    @classmethod
    def normalize_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure genre weights sum to ~1.0."""
        if not v:
            return v
        total = sum(v.values())
        if total > 0 and abs(total - 1.0) > 0.01:
            return {k: val / total for k, val in v.items()}
        return v


class DJSetRequest(BaseModel):
    """
    Main request model: the complete data contract for playlist generation.

    This is what the NLP parser outputs and the playlist generator consumes.
    """

    target_duration_minutes: int = Field(
        default=60,
        ge=5,
        le=480,
        description="Target playlist duration in minutes"
    )

    bpm_range: Tuple[int, int] = Field(
        default=(120, 130),
        description="BPM bounds [min, max] for track selection"
    )

    energy_range: Tuple[float, float] = Field(
        default=(0.5, 0.7),
        description="Energy bounds [min, max] on 0.0-1.0 scale. Chill=[0.2,0.5], Peak=[0.7,0.95]"
    )

    energy_profile: EnergyProfile = Field(
        default=EnergyProfile.STEADY_HIGH,
        description="Energy curve shape over the playlist duration"
    )

    specifics: Specifics = Field(
        default_factory=Specifics,
        description="Explicit artist/track inclusions and exclusions"
    )

    vibe: VibeProfile = Field(
        default_factory=VibeProfile,
        description="Fuzzy vibe profile for semantic matching"
    )

    reasoning: str = Field(
        default="",
        description="LLM explanation of why these constraints were chosen"
    )

    venue_style: Optional[VenueStyle] = Field(
        default=None,
        description="Venue archetype affecting track selection (inferred from keywords)"
    )

    venue_energy_modifier: float = Field(
        default=1.0,
        ge=0.5,
        le=1.5,
        description="Energy modifier based on venue (Berghain=0.85, Festival=1.1)"
    )

    # Duration constraints for track filtering
    min_duration_minutes: Optional[float] = Field(
        default=None,
        description="Minimum track duration in minutes (e.g., 8 for bathroom breaks)"
    )

    max_duration_minutes: Optional[float] = Field(
        default=None,
        description="Maximum track duration in minutes (e.g., 3 for short tracks)"
    )

    # BPM strictness
    bpm_strict: bool = Field(
        default=False,
        description="If True, filter BPM +/- 2 instead of +/- 10%"
    )

    # Target track count for pool generation
    target_tracks: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Target number of tracks for pool generation (1-500)"
    )

    # Raw query string for context
    raw_query: str = Field(
        default="",
        description="Original user query for context-aware filtering"
    )

    # Remix-type intent. Populated by the parser when the user says "no
    # remixes" / "originals only" / "remixes of popular songs" / etc. The
    # hard filter consults `gemini_remix_type` per track (original / remix
    # / edit / vip / mashup, populated by the deterministic regex backfill
    # at enrichment time). Null `gemini_remix_type` rows are treated as
    # unknown and PASS both directions of the filter — see _hard_filter.
    remix_filter: Optional[Literal["only_remixes", "no_remixes"]] = Field(
        default=None,
        description=(
            "'only_remixes' to keep only remix/edit/vip/mashup; "
            "'no_remixes' to keep only originals; null = no constraint."
        ),
    )

    # Release year constraints
    release_year_min: Optional[int] = Field(
        default=None,
        ge=1950,
        le=2030,
        description="Minimum release year (e.g., 2010 for 'after 2010')"
    )

    release_year_max: Optional[int] = Field(
        default=None,
        ge=1950,
        le=2030,
        description="Maximum release year (e.g., 2012 for 'before 2012')"
    )

    @field_validator('bpm_range')
    @classmethod
    def validate_bpm_range(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Ensure BPM range is valid."""
        min_bpm, max_bpm = v
        if min_bpm < 60:
            min_bpm = 60
        if max_bpm > 200:
            max_bpm = 200
        if min_bpm > max_bpm:
            min_bpm, max_bpm = max_bpm, min_bpm
        return (min_bpm, max_bpm)

    @field_validator('energy_range')
    @classmethod
    def validate_energy_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        """Ensure energy range is valid 0.0-1.0."""
        min_e, max_e = v
        min_e = max(0.0, min(1.0, min_e))
        max_e = max(0.0, min(1.0, max_e))
        if min_e > max_e:
            min_e, max_e = max_e, min_e
        return (min_e, max_e)

    def estimate_track_count(self, avg_track_duration_minutes: float = 5.0) -> int:
        """Estimate number of tracks needed for target duration."""
        return max(1, int(self.target_duration_minutes / avg_track_duration_minutes))

    def get_energy_at_position(self, position: float) -> float:
        """
        Get target energy at a given playlist position (0.0 to 1.0).

        Args:
            position: Position in playlist (0.0 = start, 1.0 = end)

        Returns:
            Target energy level at that position
        """
        min_e, max_e = self.energy_range
        mid_e = (min_e + max_e) / 2

        if self.energy_profile == EnergyProfile.RAMP_UP:
            return min_e + (max_e - min_e) * position

        elif self.energy_profile == EnergyProfile.RAMP_DOWN:
            return max_e - (max_e - min_e) * position

        elif self.energy_profile == EnergyProfile.MOUNTAIN:
            # Peak at 0.5
            if position < 0.5:
                return min_e + (max_e - min_e) * (position * 2)
            else:
                return max_e - (max_e - min_e) * ((position - 0.5) * 2)

        elif self.energy_profile == EnergyProfile.VALLEY:
            # Dip at 0.5
            if position < 0.5:
                return max_e - (max_e - min_e) * (position * 2)
            else:
                return min_e + (max_e - min_e) * ((position - 0.5) * 2)

        elif self.energy_profile == EnergyProfile.STEADY_HIGH:
            return max_e

        elif self.energy_profile == EnergyProfile.STEADY_LOW:
            return min_e

        elif self.energy_profile == EnergyProfile.WAVE:
            # Two peaks at 0.25 and 0.75
            import math
            return mid_e + (max_e - mid_e) * math.sin(position * 4 * math.pi)

        return mid_e

    def get_modified_energy_at_position(self, position: float) -> float:
        """
        Get venue-modified target energy at a given playlist position.

        Applies the venue_energy_modifier to adjust for venue-specific expectations.
        Berghain (0.85) = darker, more restrained peaks
        Festival (1.1) = higher energy ceiling

        Args:
            position: Position in playlist (0.0 = start, 1.0 = end)

        Returns:
            Venue-adjusted target energy level
        """
        base_energy = self.get_energy_at_position(position)
        modified = base_energy * self.venue_energy_modifier
        return max(0.0, min(1.0, modified))

    def summary(self) -> str:
        """Human-readable summary of the request."""
        lines = [
            f"Duration: {self.target_duration_minutes} min (~{self.estimate_track_count()} tracks)",
            f"BPM: {self.bpm_range[0]}-{self.bpm_range[1]}",
            f"Energy: {self.energy_range[0]:.1f}-{self.energy_range[1]:.1f} ({self.energy_profile.value})",
        ]

        if self.venue_style:
            lines.append(f"Venue: {self.venue_style.value} (modifier: {self.venue_energy_modifier:.2f})")

        if self.vibe.keywords:
            lines.append(f"Vibes: {', '.join(self.vibe.keywords)}")

        if self.vibe.genre_weights:
            genres = [f"{g}:{w:.0%}" for g, w in self.vibe.genre_weights.items()]
            lines.append(f"Genres: {', '.join(genres)}")

        if self.specifics.included_artists:
            lines.append(f"Include: {', '.join(self.specifics.included_artists)}")

        if self.specifics.excluded_artists:
            lines.append(f"Exclude: {', '.join(self.specifics.excluded_artists)}")

        return "\n".join(lines)

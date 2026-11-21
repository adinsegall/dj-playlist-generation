"""
Publication-Quality Visualizations for DJ Playlist Evaluation.

Generates figures for:
- t-SNE/UMAP embeddings colored by genre
- Confusion matrices
- Ablation bar charts
- Distribution plots
- Energy curves
- Metric comparison tables

Author: Adin Segall
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Custom color palette for genres
GENRE_COLORS = {
    'techno': '#FF4136',
    'tech house': '#FF851B',
    'deep house': '#2ECC40',
    'progressive house': '#0074D9',
    'house': '#B10DC9',
    'trance': '#01FF70',
    'melodic techno': '#F012BE',
    'drum and bass': '#FFDC00',
    'dubstep': '#7FDBFF',
    'disco': '#FF69B4',
    'minimal': '#85144b',
    'electronica': '#39CCCC',
}

PROJECT_ROOT = Path(__file__).parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "evaluation" / "figures"


class EmbeddingVisualizer:
    """Visualize artist embeddings using dimensionality reduction."""

    def __init__(self, model=None, artist_genres: Optional[Dict[str, str]] = None):
        """
        Initialize visualizer.

        Args:
            model: Word2Vec model
            artist_genres: Dict mapping artist to genre
        """
        self.model = model
        self.wv = model.wv if model else None
        self.artist_genres = artist_genres or {}

        # Create figures directory
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def plot_tsne_embeddings(
        self,
        n_artists: int = 2000,
        top_genres: int = 10,
        perplexity: int = 30,
        seed: int = 42,
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot t-SNE visualization of artist embeddings colored by genre.

        Args:
            n_artists: Number of artists to plot
            top_genres: Number of top genres to include
            perplexity: t-SNE perplexity parameter
            seed: Random seed
            save_path: Path to save figure
            show: Whether to display the figure

        Returns:
            Matplotlib Figure object
        """
        if not self.wv:
            print("No model loaded")
            return None

        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("sklearn required for t-SNE")
            return None

        # Get artists with known genres
        artists_with_genre = [
            (a, g) for a, g in self.artist_genres.items()
            if a.lower() in self.wv
        ]

        if not artists_with_genre:
            print("No artists with genre information found in model")
            return None

        # Filter to top genres
        genre_counts = Counter(g for a, g in artists_with_genre)
        top_genre_names = [g for g, c in genre_counts.most_common(top_genres)]

        artists_filtered = [
            (a, g) for a, g in artists_with_genre
            if g in top_genre_names
        ]

        # Sample if too many
        np.random.seed(seed)
        if len(artists_filtered) > n_artists:
            indices = np.random.choice(len(artists_filtered), n_artists, replace=False)
            artists_filtered = [artists_filtered[i] for i in indices]

        # Extract embeddings
        artists = [a for a, g in artists_filtered]
        genres = [g for a, g in artists_filtered]
        embeddings = np.array([self.wv[a.lower()] for a in artists])

        print(f"Running t-SNE on {len(artists)} artists...")

        # Run t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=seed,
            n_iter=1000
        )
        coords = tsne.fit_transform(embeddings)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))

        # Create color map
        unique_genres = sorted(set(genres))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_genres)))
        genre_to_color = {g: colors[i] for i, g in enumerate(unique_genres)}

        # Plot each genre separately for legend
        for genre in unique_genres:
            mask = np.array([g == genre for g in genres])
            color = genre_to_color.get(genre, 'gray')
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[color],
                label=genre,
                alpha=0.6,
                s=30
            )

        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('Artist Embeddings by Genre (t-SNE)')
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            frameon=True,
            fontsize=10
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_umap_embeddings(
        self,
        n_artists: int = 2000,
        top_genres: int = 10,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        seed: int = 42,
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot UMAP visualization of artist embeddings.

        Args:
            n_artists: Number of artists to plot
            top_genres: Number of top genres
            n_neighbors: UMAP neighbors parameter
            min_dist: UMAP min_dist parameter
            seed: Random seed
            save_path: Path to save figure
            show: Whether to display

        Returns:
            Matplotlib Figure object
        """
        if not self.wv:
            print("No model loaded")
            return None

        try:
            import umap
        except ImportError:
            print("umap-learn required for UMAP visualization")
            print("Install with: pip install umap-learn")
            return None

        # Get artists with known genres
        artists_with_genre = [
            (a, g) for a, g in self.artist_genres.items()
            if a.lower() in self.wv
        ]

        if not artists_with_genre:
            print("No artists with genre information found in model")
            return None

        # Filter to top genres
        genre_counts = Counter(g for a, g in artists_with_genre)
        top_genre_names = [g for g, c in genre_counts.most_common(top_genres)]

        artists_filtered = [
            (a, g) for a, g in artists_with_genre
            if g in top_genre_names
        ]

        # Sample if too many
        np.random.seed(seed)
        if len(artists_filtered) > n_artists:
            indices = np.random.choice(len(artists_filtered), n_artists, replace=False)
            artists_filtered = [artists_filtered[i] for i in indices]

        # Extract embeddings
        artists = [a for a, g in artists_filtered]
        genres = [g for a, g in artists_filtered]
        embeddings = np.array([self.wv[a.lower()] for a in artists])

        print(f"Running UMAP on {len(artists)} artists...")

        # Run UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed,
            n_components=2
        )
        coords = reducer.fit_transform(embeddings)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot
        unique_genres = sorted(set(genres))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_genres)))
        genre_to_color = {g: colors[i] for i, g in enumerate(unique_genres)}

        for genre in unique_genres:
            mask = np.array([g == genre for g in genres])
            color = genre_to_color.get(genre, 'gray')
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[color],
                label=genre,
                alpha=0.6,
                s=30
            )

        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        ax.set_title('Artist Embeddings by Genre (UMAP)')
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            frameon=True,
            fontsize=10
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        return fig


class MetricVisualizer:
    """Visualize evaluation metrics and comparisons."""

    def __init__(self):
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(
        self,
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None,
        title: str = "Genre Classification Confusion Matrix",
        save_path: Optional[Path] = None,
        show: bool = True,
        normalize: bool = True
    ) -> plt.Figure:
        """
        Plot confusion matrix for genre classification.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label order (if None, sorted unique)
            title: Plot title
            save_path: Path to save
            show: Whether to display
            normalize: Whether to normalize by row

        Returns:
            Matplotlib Figure
        """
        from sklearn.metrics import confusion_matrix

        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

        fig, ax = plt.subplots(figsize=(12, 10))

        # Use seaborn heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )

        ax.set_xlabel('Predicted Genre')
        ax.set_ylabel('True Genre')
        ax.set_title(title)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_ablation_bar_chart(
        self,
        ablation_results: Dict[str, Dict[str, float]],
        metric_name: str = "Coherence",
        title: str = "Ablation Study Results",
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot bar chart comparing ablation conditions.

        Args:
            ablation_results: Dict of {condition_name: {metric: value, ci_lower: val, ci_upper: val}}
            metric_name: Which metric to plot
            title: Plot title
            save_path: Path to save
            show: Whether to display

        Returns:
            Matplotlib Figure
        """
        conditions = list(ablation_results.keys())
        values = [ablation_results[c].get(metric_name, 0) for c in conditions]

        # Get confidence intervals if available
        errors = []
        for c in conditions:
            ci_lower = ablation_results[c].get('ci_lower', 0)
            ci_upper = ablation_results[c].get('ci_upper', 0)
            value = ablation_results[c].get(metric_name, 0)
            if ci_lower > 0 and ci_upper > 0:
                errors.append([[value - ci_lower], [ci_upper - value]])
            else:
                errors.append([[0], [0]])

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(conditions))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(conditions)))

        bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=1)

        # Add error bars if available
        for i, (err, bar) in enumerate(zip(errors, bars)):
            if err[0][0] > 0 or err[1][0] > 0:
                ax.errorbar(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    yerr=[[err[0][0]], [err[1][0]]],
                    fmt='none',
                    color='black',
                    capsize=5
                )

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=11
            )

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_ylabel(metric_name)
        ax.set_title(title)

        # Add baseline reference line (first condition)
        if values:
            ax.axhline(y=values[0], color='red', linestyle='--', alpha=0.5, label='Baseline')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_mode_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        metrics: List[str] = ['coherence', 'genre_entropy', 'transition_quality'],
        title: str = "Mode Comparison",
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot comparison of PRO/CASUAL/HYBRID modes across metrics.

        Args:
            results: Dict of {mode: {metric: {mean, ci_lower, ci_upper}}}
            metrics: List of metric names to plot
            title: Plot title
            save_path: Path to save
            show: Whether to display

        Returns:
            Matplotlib Figure
        """
        modes = list(results.keys())
        n_metrics = len(metrics)
        n_modes = len(modes)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(n_metrics)
        width = 0.8 / n_modes

        colors = ['#2ECC40', '#FF851B', '#0074D9']  # Green, Orange, Blue

        for i, mode in enumerate(modes):
            means = []
            errors_lower = []
            errors_upper = []

            for metric in metrics:
                metric_data = results[mode].get(metric, {})
                mean = metric_data.get('mean', 0)
                ci_lower = metric_data.get('ci_lower', mean)
                ci_upper = metric_data.get('ci_upper', mean)

                means.append(mean)
                errors_lower.append(mean - ci_lower)
                errors_upper.append(ci_upper - mean)

            offset = (i - n_modes/2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                means,
                width,
                label=mode,
                color=colors[i % len(colors)],
                edgecolor='black',
                linewidth=0.5
            )

            # Add error bars
            ax.errorbar(
                x + offset,
                means,
                yerr=[errors_lower, errors_upper],
                fmt='none',
                color='black',
                capsize=3
            )

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_transition_distribution(
        self,
        source_counts: Dict[str, int],
        title: str = "Transition Source Distribution",
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot pie/bar chart of transition sources.

        Args:
            source_counts: Dict of {source: count}
            title: Plot title
            save_path: Path to save
            show: Whether to display

        Returns:
            Matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        sources = list(source_counts.keys())
        counts = list(source_counts.values())
        total = sum(counts)
        percentages = [c / total * 100 for c in counts]

        colors = plt.cm.Set2(np.linspace(0, 1, len(sources)))

        # Pie chart
        ax1.pie(
            counts,
            labels=sources,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax1.set_title(f'{title} (Pie)')

        # Bar chart
        bars = ax2.bar(sources, percentages, color=colors, edgecolor='black')

        for bar, pct in zip(bars, percentages):
            ax2.annotate(
                f'{pct:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom'
            )

        ax2.set_ylabel('Percentage')
        ax2.set_title(f'{title} (Bar)')
        ax2.set_ylim(0, max(percentages) * 1.2)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_energy_curve(
        self,
        playlist: List[Dict[str, Any]],
        title: str = "Playlist Energy Curve",
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot energy curve for a playlist.

        Args:
            playlist: List of track dicts with energy values
            title: Plot title
            save_path: Path to save
            show: Whether to display

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        positions = list(range(1, len(playlist) + 1))
        estimated = [t.get('estimated_energy', 0.5) for t in playlist]
        target = [t.get('target_energy', 0.5) for t in playlist]

        ax.plot(positions, estimated, 'b-o', label='Estimated Energy', linewidth=2, markersize=8)
        ax.plot(positions, target, 'r--', label='Target Energy', linewidth=2, alpha=0.7)

        # Fill between
        ax.fill_between(positions, estimated, target, alpha=0.2, color='gray')

        ax.set_xlabel('Track Position')
        ax.set_ylabel('Energy Level')
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_xlim(0.5, len(playlist) + 0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_metric_comparison_table(
        self,
        comparison_results: Dict[str, Dict[str, Any]],
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create a table visualization of metric comparisons.

        Args:
            comparison_results: Output from StatisticalTests.compare_approaches()
            save_path: Path to save
            show: Whether to display

        Returns:
            Matplotlib Figure
        """
        # Prepare data for table
        rows = []
        for metric_name, data in comparison_results.items():
            row = {
                'Metric': metric_name.replace('_', ' ').title(),
                'PRO Mode': f"{data['approach_a']['mean']:.3f} +/- {data['approach_a']['std']:.3f}",
                'CASUAL Mode': f"{data['approach_b']['mean']:.3f} +/- {data['approach_b']['std']:.3f}",
                'p-value': f"{data['wilcoxon_p']:.4f}",
                "Cohen's d": f"{data['effect_size']:.3f}",
                'Significant': 'Yes' if data['significant'] else 'No'
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.axis('off')

        # Create table
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#4472C4'] * len(df.columns)
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Color header
        for i, key in enumerate(df.columns):
            table[0, i].set_text_props(color='white', weight='bold')

        # Color significant rows
        for i, row in df.iterrows():
            if row['Significant'] == 'Yes':
                for j in range(len(df.columns)):
                    table[i + 1, j].set_facecolor('#E6FFE6')

        plt.title('Metric Comparison: PRO vs CASUAL Mode', fontsize=14, pad=20)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_source_similarity_heatmap(
        self,
        source_analysis: Dict[str, Any],
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot heatmap of source tag similarities.

        Args:
            source_analysis: Output from EmbeddingMetrics.source_tag_analysis()
            save_path: Path to save
            show: Whether to display

        Returns:
            Matplotlib Figure
        """
        sources = ['SOURCE_1001TL', 'SOURCE_DJMIX', 'SOURCE_MPD']
        n = len(sources)

        # Build similarity matrix
        sim_matrix = np.eye(n)

        for i, src_a in enumerate(sources):
            for j, src_b in enumerate(sources):
                if i < j:
                    key = f"{src_a}_vs_{src_b}"
                    if key in source_analysis:
                        sim_matrix[i, j] = source_analysis[key]
                        sim_matrix[j, i] = source_analysis[key]

        fig, ax = plt.subplots(figsize=(8, 6))

        labels = ['1001TL', 'DJMix', 'MPD']

        sns.heatmap(
            sim_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            vmin=-1,
            vmax=1,
            center=0
        )

        ax.set_title('Source Tag Similarity Matrix')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        return fig


class SurveyVisualizer:
    """Visualize user study survey results."""

    def __init__(self):
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def plot_likert_responses(
        self,
        likert_data: Dict[str, Dict[str, Any]],
        title: str = "Survey Likert Scale Responses",
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot horizontal bar chart of Likert scale responses.

        Args:
            likert_data: Dict of {question: {mean, std, n, distribution}}
            title: Plot title
            save_path: Path to save
            show: Whether to display

        Returns:
            Matplotlib Figure
        """
        questions = list(likert_data.keys())
        means = [likert_data[q]['mean'] for q in questions]
        stds = [likert_data[q]['std'] for q in questions]

        # Shorten questions for display
        short_questions = []
        for q in questions:
            if len(q) > 45:
                short_questions.append(q[:42] + "...")
            else:
                short_questions.append(q)

        fig, ax = plt.subplots(figsize=(12, 10))

        y_pos = np.arange(len(questions))

        # Color bars by score
        colors = plt.cm.RdYlGn(np.array(means) / 5.0)

        bars = ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor='black',
                       capsize=3, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_questions)
        ax.set_xlabel('Mean Score (1-5)')
        ax.set_xlim(1, 5)
        ax.set_title(title)

        # Add vertical lines for reference
        for x in [2, 3, 4]:
            ax.axvline(x=x, color='gray', linestyle='--', alpha=0.3)

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(mean + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{mean:.2f}', va='center', fontsize=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_system_handling_radar(
        self,
        handling_data: Dict[str, Dict[str, Any]],
        title: str = "System Handling Assessment",
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot radar chart of system handling scores.

        Args:
            handling_data: Dict of {category: {mean, std}}
            title: Plot title
            save_path: Path to save
            show: Whether to display

        Returns:
            Matplotlib Figure
        """
        categories = list(handling_data.keys())
        values = [handling_data[c]['mean'] for c in categories]

        # Format labels
        labels = [c.replace('_', ' ').title() for c in categories]

        # Number of variables
        N = len(categories)

        # Compute angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the loop

        values += values[:1]  # Complete the loop

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Draw the polygon
        ax.plot(angles, values, 'o-', linewidth=2, color='#0074D9')
        ax.fill(angles, values, alpha=0.25, color='#0074D9')

        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=12)

        # Set y-axis limits (1-5 scale)
        ax.set_ylim(1, 5)
        ax.set_yticks([2, 3, 4, 5])
        ax.set_yticklabels(['2', '3', '4', '5'])

        ax.set_title(title, size=14, y=1.1)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_satisfaction_distribution(
        self,
        satisfaction_data: Dict[str, Any],
        title: str = "Playlist Satisfaction Distribution",
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot histogram of satisfaction ratings.

        Args:
            satisfaction_data: Dict with 'distribution' key mapping scores to counts
            title: Plot title
            save_path: Path to save
            show: Whether to display

        Returns:
            Matplotlib Figure
        """
        distribution = satisfaction_data.get('distribution', {})
        mean = satisfaction_data.get('mean', 0)
        std = satisfaction_data.get('std', 0)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Ensure all ratings 1-5 are represented
        ratings = [1, 2, 3, 4, 5]
        counts = [distribution.get(r, distribution.get(str(r), 0)) for r in ratings]

        colors = plt.cm.RdYlGn(np.array(ratings) / 5.0)

        bars = ax.bar(ratings, counts, color=colors, edgecolor='black', width=0.7)

        ax.set_xlabel('Satisfaction Rating')
        ax.set_ylabel('Count')
        ax.set_title(f'{title}\nMean: {mean:.2f} (SD: {std:.2f})')
        ax.set_xticks(ratings)
        ax.set_xticklabels(['1\n(Very Low)', '2\n(Low)', '3\n(Neutral)',
                           '4\n(High)', '5\n(Very High)'])

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontsize=12)

        # Add mean line
        ax.axvline(x=mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
        ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_generation_time_comparison(
        self,
        time_data: Dict[str, int],
        title: str = "Generation Time vs Manual Creation",
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot bar chart of generation time comparison responses.

        Args:
            time_data: Dict of {response: count}
            title: Plot title
            save_path: Path to save
            show: Whether to display

        Returns:
            Matplotlib Figure
        """
        # Ensure ordered responses
        order = ['Much slower', 'Somewhat slower', 'About the same',
                 'Somewhat faster', 'Much faster']
        responses = [r for r in order if r in time_data]
        counts = [time_data.get(r, 0) for r in responses]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Color gradient from red to green
        colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(responses)))

        bars = ax.bar(responses, counts, color=colors, edgecolor='black')

        ax.set_xlabel('Response')
        ax.set_ylabel('Count')
        ax.set_title(title)

        plt.xticks(rotation=15, ha='right')

        # Add count labels
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontsize=11)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_quality_metrics(
        self,
        quality_data: Dict[str, Any],
        title: str = "Playlist Quality Assessment",
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot quality metrics (% appropriate, % reordering).

        Args:
            quality_data: Dict with 'pct_appropriate' and 'pct_reordering' sub-dicts
            title: Plot title
            save_path: Path to save
            show: Whether to display

        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # % Appropriate
        if 'pct_appropriate' in quality_data:
            ax = axes[0]
            data = quality_data['pct_appropriate']
            mean = data.get('mean', 0)
            std = data.get('std', 0)

            ax.bar(['% Appropriate'], [mean], yerr=[std], color='#2ECC40',
                   edgecolor='black', capsize=10, width=0.5)
            ax.set_ylabel('Percentage')
            ax.set_title(f'Tracks Appropriate for Prompt\nMean: {mean:.1f}%')
            ax.set_ylim(0, 100)
            ax.axhline(y=70, color='gray', linestyle='--', alpha=0.5, label='70% threshold')
            ax.legend()

        # % Reordering needed
        if 'pct_reordering' in quality_data:
            ax = axes[1]
            data = quality_data['pct_reordering']
            mean = data.get('mean', 0)
            std = data.get('std', 0)

            ax.bar(['% Needing Reorder'], [mean], yerr=[std], color='#FF851B',
                   edgecolor='black', capsize=10, width=0.5)
            ax.set_ylabel('Percentage')
            ax.set_title(f'Tracks Requiring Reordering\nMean: {mean:.1f}%')
            ax.set_ylim(0, 100)

        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()

        return fig

    def generate_all_survey_figures(
        self,
        stats: Dict[str, Any],
        output_dir: Optional[Path] = None,
        show: bool = False
    ) -> Dict[str, Path]:
        """
        Generate all survey visualization figures.

        Args:
            stats: Output from UserStudyAnalyzer.compute_statistics()
            output_dir: Directory to save figures
            show: Whether to display figures

        Returns:
            Dict mapping figure names to paths
        """
        output_dir = output_dir or FIGURES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        # Likert responses
        if stats.get('likert_scores'):
            path = output_dir / "survey_likert_responses.png"
            self.plot_likert_responses(
                stats['likert_scores'],
                save_path=path,
                show=show
            )
            paths['likert'] = path

        # System handling radar
        if stats.get('system_handling'):
            path = output_dir / "survey_system_handling.png"
            self.plot_system_handling_radar(
                stats['system_handling'],
                save_path=path,
                show=show
            )
            paths['system_handling'] = path

        # Satisfaction distribution
        if stats.get('satisfaction'):
            path = output_dir / "survey_satisfaction.png"
            self.plot_satisfaction_distribution(
                stats['satisfaction'],
                save_path=path,
                show=show
            )
            paths['satisfaction'] = path

        # Generation time
        if stats.get('quality_assessment', {}).get('generation_time_comparison'):
            path = output_dir / "survey_generation_time.png"
            self.plot_generation_time_comparison(
                stats['quality_assessment']['generation_time_comparison'],
                save_path=path,
                show=show
            )
            paths['generation_time'] = path

        # Quality metrics
        if stats.get('quality_assessment'):
            path = output_dir / "survey_quality_metrics.png"
            self.plot_quality_metrics(
                stats['quality_assessment'],
                save_path=path,
                show=show
            )
            paths['quality_metrics'] = path

        print(f"\nGenerated {len(paths)} survey figures in {output_dir}")
        return paths


def save_results_json(
    results: Dict[str, Any],
    path: Path = PROJECT_ROOT / "evaluation" / "results.json"
):
    """Save results to JSON for dissertation tables."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results_converted = convert(results)

    with open(path, 'w') as f:
        json.dump(results_converted, f, indent=2)

    print(f"Results saved to {path}")


if __name__ == "__main__":
    """Quick test of visualizations."""
    print("=" * 60)
    print("VISUALIZATION TEST")
    print("=" * 60)

    # Create test data
    comparison_results = {
        'coherence': {
            'approach_a': {'mean': 0.82, 'std': 0.05, 'ci_lower': 0.79, 'ci_upper': 0.85},
            'approach_b': {'mean': 0.65, 'std': 0.07, 'ci_lower': 0.61, 'ci_upper': 0.69},
            'wilcoxon_p': 0.001,
            'effect_size': 2.8,
            'significant': True
        },
        'genre_entropy': {
            'approach_a': {'mean': 0.23, 'std': 0.04, 'ci_lower': 0.21, 'ci_upper': 0.25},
            'approach_b': {'mean': 0.45, 'std': 0.06, 'ci_lower': 0.42, 'ci_upper': 0.48},
            'wilcoxon_p': 0.002,
            'effect_size': -3.5,
            'significant': True
        },
        'transition_quality': {
            'approach_a': {'mean': 0.71, 'std': 0.08, 'ci_lower': 0.67, 'ci_upper': 0.75},
            'approach_b': {'mean': 0.43, 'std': 0.09, 'ci_lower': 0.38, 'ci_upper': 0.48},
            'wilcoxon_p': 0.001,
            'effect_size': 3.2,
            'significant': True
        }
    }

    viz = MetricVisualizer()

    # Test table
    print("\nGenerating comparison table...")
    viz.plot_metric_comparison_table(
        comparison_results,
        save_path=FIGURES_DIR / "test_comparison_table.png",
        show=False
    )

    # Test transition distribution
    print("\nGenerating transition distribution...")
    source_counts = {
        'transition': 45,
        'proxy': 15,
        'embedding': 30,
        'constraint': 10
    }
    viz.plot_transition_distribution(
        source_counts,
        save_path=FIGURES_DIR / "test_transition_dist.png",
        show=False
    )

    # Test energy curve
    print("\nGenerating energy curve...")
    test_playlist = [
        {'estimated_energy': 0.4, 'target_energy': 0.4},
        {'estimated_energy': 0.45, 'target_energy': 0.48},
        {'estimated_energy': 0.52, 'target_energy': 0.55},
        {'estimated_energy': 0.58, 'target_energy': 0.62},
        {'estimated_energy': 0.65, 'target_energy': 0.70},
        {'estimated_energy': 0.72, 'target_energy': 0.75},
        {'estimated_energy': 0.78, 'target_energy': 0.80},
        {'estimated_energy': 0.82, 'target_energy': 0.82},
        {'estimated_energy': 0.80, 'target_energy': 0.78},
        {'estimated_energy': 0.75, 'target_energy': 0.72},
    ]
    viz.plot_energy_curve(
        test_playlist,
        title="Test Energy Curve",
        save_path=FIGURES_DIR / "test_energy_curve.png",
        show=False
    )

    print(f"\nFigures saved to {FIGURES_DIR}")
    print("\n" + "=" * 60)
    print("VISUALIZATION TEST COMPLETE")
    print("=" * 60)

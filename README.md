# gAItKeep: AEI Interactive Data Explorer

An interactive visualization platform for exploring the **Anthropic Economic Index (AEI)** dataset. Discover patterns in AI task competition, inequality metrics, and labor market dynamics with drag-and-drop chart building.

## Live Demo

**[Launch the Interactive Explorer →](https://pranavmehta-git.github.io/gaitkeep/home.html)**

## Features

### 🔍 Interactive Data Explorer
A Tableau-style tool for exploring the AEI data:
- **Dimension Selection**: Choose X-axis (Occupation, SOC Group, Task Type, Year)
- **Metric Selection**: Mean/Median Incumbents, Gini, Count, CV, and more
- **Chart Types**: Bar, Horizontal Bar, Line, and Scatter plots
- **Preset Analyses**: One-click access to meaningful comparisons
- **Dynamic Insights**: Auto-generated findings based on your selection

### 📈 Competition Paradox Dashboard
Deep-dive analysis of inequality in task competition:
- **Lorenz Curves**: Visualize competition distribution
- **Gini Coefficient**: Measure inequality (0.27 across all tasks)
- **Regression Analysis**: Diminishing returns in competition (R² = 0.77)
- **HHI Metrics**: Market concentration analysis

## Quick Start

### View the Dashboards
No installation needed! Visit the [live demo](https://pranavmehta-git.github.io/gaitkeep/home.html).

### Run Locally
```bash
# Clone the repository
git clone https://github.com/pranavmehta-git/gaitkeep.git
cd gaitkeep

# Serve the dashboard locally
python -m http.server 8000 -d results/dashboard

# Open http://localhost:8000/home.html
```

### Regenerate Data
```bash
# Install dependencies
pip install -r requirements.txt

# Generate interactive data
python src/generate_interactive_data.py
```

## Available Analyses

| Preset | Description |
|--------|-------------|
| Inequality by Occupation | Compare Gini coefficients across jobs |
| Competition by Industry | Mean incumbents by SOC major group |
| Yearly Trends | Temporal evolution of competition |
| Core vs Supplemental | Compare task types |
| Task Volume by Industry | Which sectors have the most tasks? |
| Inequality Over Time | Gini trends by year |

## Data Overview

| Metric | Value |
|--------|-------|
| Total Records | 18,872 |
| Unique Occupations | 951 |
| SOC Major Groups | 22 |
| Time Range | 2003-2015 |
| Overall Gini | 0.2746 |
| Core Tasks | 71.5% |

## Project Structure

```
gaitkeep/
├── results/dashboard/          # Interactive visualizations
│   ├── home.html              # Landing page
│   ├── explorer.html          # Interactive data explorer
│   ├── index.html             # Competition paradox dashboard
│   ├── explorer_data.json     # Pre-computed metrics
│   └── interactive_data.json  # Full interactive dataset
├── src/
│   ├── generate_interactive_data.py  # Data generator for explorer
│   ├── exploratory_analysis.py       # EDA and inequality metrics
│   ├── regression_analysis.py        # Statistical modeling
│   └── ...
├── configs/
└── requirements.txt
```

## Key Findings

1. **Moderate Inequality**: Gini coefficient of 0.27 indicates moderate concentration
2. **Industry Variation**: Office/Admin roles have highest competition (~109 mean incumbents); Computer/Math lowest (~48)
3. **Task Type Difference**: Supplemental tasks attract 17% more incumbents than Core tasks
4. **Diminishing Returns**: Competition benefit diminishes logarithmically after ~80 incumbents
5. **Temporal Stability**: Inequality patterns remain relatively stable over 12 years

## Data Source

**[Anthropic Economic Index](https://huggingface.co/datasets/Anthropic/EconomicIndex)** - Measures AI task competition patterns across occupations, mapping tasks to O*NET Standard Occupational Classification.

## Technologies

- **D3.js** - Interactive visualizations
- **Python** - Data processing (pandas, numpy)
- **GitHub Pages** - Hosting

## License

MIT License

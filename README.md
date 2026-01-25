# GaitKeep: AEI Dataset Analysis

Analysis pipeline for exploring the Anthropic Economic Index (AEI) dataset to uncover patterns related to AI usage and economic inequality.

## Project Structure

```
gaitkeep/
├── data/                    # Data directory (gitignored)
│   ├── raw/                 # Original downloaded data
│   └── processed/           # Cleaned, analysis-ready data
├── src/                     # Source code
│   ├── data_ingestion.py    # Download and load AEI dataset
│   ├── data_cleaning.py     # Preprocessing and cleaning
│   ├── exploratory_analysis.py  # EDA and inequality metrics
│   ├── regression_analysis.py   # Statistical modeling
│   └── visualization_outputs.py # Dashboard-ready outputs
├── results/                 # Analysis outputs
│   ├── figures/             # Generated plots
│   └── tables/              # Summary tables
├── configs/                 # Configuration files
├── requirements.txt         # Python dependencies
└── README.md
```

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Ingestion
Download and load the AEI dataset:
```bash
python -m src.data_ingestion
```

### 2. Data Cleaning
Preprocess the data:
```bash
python -m src.data_cleaning
```

### 3. Exploratory Analysis
Run exploratory analysis:
```bash
python -m src.exploratory_analysis
```

### 4. Regression Analysis
Run regression models:
```bash
python -m src.regression_analysis
```

### 5. Full Pipeline
Run the complete pipeline:
```bash
python -m src.run_pipeline
```

## Key Analyses

- **Task usage distribution** by income decile or occupation class
- **AI success rate** by education level or task complexity
- **Usage intensity** by region (normalized per capita)
- **Collaboration mode** analysis (automation vs. augmentation)
- **Inequality metrics**: Gini coefficients, Theil index on AI usage

## Data Sources

- **Primary**: [Anthropic Economic Index](https://huggingface.co/datasets/Anthropic/EconomicIndex)
- **Optional enrichment**: O*NET, ACS, World Bank data

## License

MIT License

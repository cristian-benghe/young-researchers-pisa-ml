# Predicting Top Academic Performers Using PISA 2022 Data

A beginner-friendly machine learning project that uses a Decision Tree to predict whether a student belongs to the **top 10% of academic performers in their country**, based on background, learning environment, and study habits.

## Research Question

> Can information about a student's background, learning environment, and study habits be used to predict whether they belong to the top 10% of academic performers in their country?

## Dataset

This project uses the official **OECD PISA 2022** student-level dataset. PISA (Programme for International Student Assessment) tests 15-year-old students worldwide in mathematics, reading, and science every three years.

We focus on students from **five diverse EU countries**:

| Country     | Country Code |
|-------------|:------------:|
| Romania     | ROU          |
| Germany     | DEU          |
| Finland     | FIN          |
| Netherlands | NLD          |
| Spain       | ESP          |

## Target Variable

For each country separately:
1. An `academic_score` is computed as the average of the first plausible values in mathematics, reading, and science (PV1MATH, PV1READ, PV1SCIE).
2. The 90th percentile of `academic_score` is calculated within each country.
3. `top_10_percent = 1` if the student scores above their country's 90th percentile, `0` otherwise.

The PISA test scores are used **only** to define the target. They are **not** used as input features (that would be data leakage).

## Features Used

The model uses approximately 15 interpretable features covering:
- Socioeconomic background (ESCS index, parents' education)
- Home resources (books, computer, internet, quiet study space)
- School attendance (absences, tardiness)
- Study habits and time (learning time, homework hours)
- Student attitudes (sense of belonging, growth mindset)
- Country

## Model

A single `DecisionTreeClassifier` from scikit-learn, kept interpretable with limited depth. No ensemble methods or neural networks are used.

## How to Run in Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File > Open Notebook > GitHub**
3. Paste this repository's URL
4. Open `notebooks/pisa_decision_tree.ipynb`
5. Run all cells (Runtime > Run all)

The notebook downloads and processes the PISA data automatically. The processed dataset (`data/processed/pisa_5countries.parquet`) is also included in this repo for faster loading.

## Repository Structure

```
young-researchers-pisa-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── pisa_decision_tree.ipynb    # Main notebook (run this)
├── data/
│   └── processed/
│       └── pisa_5countries.parquet # Pre-processed dataset (~5-10k students/country)
└── src/
    └── process_pisa_data.py        # Data download & processing script
```

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies
- All packages are pip-installable and available in Google Colab

## Authors

Young Researchers project, 2024-2025.

## License

This project uses publicly available OECD PISA data for educational and research purposes.

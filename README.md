# Overview

A computer vision program that recognizes finnish sign language letters (same as ASL with the addition of Ä, Ö, and Å) based on hand landmarks.

Only static letters can be recognizes, i.e. letters that don't require motion to sign.

## Documentation

- [Specification document](docs/specification_document.md)
- [Implementation document](docs/implementation_document.md)
- [Test document](docs/test_document.md)
- [Pylint report](docs/pylint_report.txt)

### Weekly reports

- [Week 1](docs/weekly_progress/week_1.md)
- [Week 2](docs/weekly_progress/week_2.md)
- [Week 3](docs/weekly_progress/week_3.md)
- [Week 4](docs/weekly_progress/week_4.md)
- [Week 5](docs/weekly_progress/week_5.md)
- [Week 6](docs/weekly_progress/week_6.md)



# Usage

First, clone the repository, navigate to the project, and unzip the dataset. Then, create the virtual environment with
```bash
poetry install
```

Finally, run the program from project root with
```bash
poetry run python -m src.main
```

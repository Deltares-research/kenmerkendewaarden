[![pytest](https://github.com/deltares-research/kenmerkendewaarden/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/deltares-research/kenmerkendewaarden/actions/workflows/pytest.yml)
[![codecov](https://img.shields.io/codecov/c/github/deltares-research/kenmerkendewaarden.svg?style=flat-square)](https://app.codecov.io/gh/deltares-research/kenmerkendewaarden?displayType=list)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Deltares-research_kenmerkendewaarden&metric=alert_status)](https://sonarcloud.io/dashboard?id=Deltares-research_kenmerkendewaarden)

# kenmerkendewaarden
Voor het afleiden van kengetallen als slotgemiddelden, gemiddelde getijkrommen, havengetallen, overschrijdingsfrequenties op basis van waterstandsmetingen. Meer informatie over Kenmerkende Waarden is beschikbaar op [rijkswaterstaat.nl](https://www.rijkswaterstaat.nl/water/waterbeheer/metingen/meten-bij-rijkswaterstaat/waternormalen)

## LET OP
De metodieken in deze repository hebben nog geen definitieve status en zijn daarom nog niet geschikt voor productie.

## installation
- open anaconda prompt
- `conda create --name kw_env python=3.11 git -y`
- `conda activate kw_env`
- `pip install git+https://github.com/Deltares-research/kenmerkendewaarden`

## contributing
- open git bash window
- `git clone https://github.com/deltares-research/kenmerkendewaarden`
- `cd kenmerkendewaarden`
- open anaconda prompt
- `conda create --name kw_env python=3.11 git spyder -c conda-forge -y`
- `conda activate kw_env`
- `pip install -e .[dev,examples]`
- more contributing guidelines available on dfm_tools repos: https://deltares.github.io/dfm_tools/CONTRIBUTING.html
